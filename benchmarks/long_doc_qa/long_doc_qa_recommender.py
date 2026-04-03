# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import math
import re
import resource
import subprocess
from typing import Any

# Third Party
from huggingface_hub import HfApi
import psutil
import torch


def determine_per_gpu_memory():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return total_memory


def _extract_parameter_count_from_model_name(model_name: str) -> int | None:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*([bBmMkK])", model_name)
    if not matches:
        return None
    magnitude, suffix = matches[-1]
    scale = {"k": 10**3, "m": 10**6, "b": 10**9}[suffix.lower()]
    return int(float(magnitude) * scale)


def _get_total_parameter_count(info: Any, model_name: str) -> int:
    safetensors = getattr(info, "safetensors", None)
    if safetensors is not None:
        parameters = getattr(safetensors, "parameters", None)
        if parameters:
            return sum(parameters.values())
        total = getattr(safetensors, "total", None)
        if total:
            return int(total)

    for attr_name in ("cardData", "config", "transformersInfo"):
        metadata = getattr(info, attr_name, None)
        if not isinstance(metadata, dict):
            continue
        for key in (
            "num_parameters",
            "parameter_count",
            "model_size",
            "num_params",
            "parameters",
        ):
            value = metadata.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, float) and value > 0:
                return int(value)
            if isinstance(value, str):
                parsed = _extract_parameter_count_from_model_name(value)
                if parsed is not None:
                    return parsed

    parsed = _extract_parameter_count_from_model_name(model_name)
    if parsed is not None:
        return parsed

    raise RuntimeError(
        "Unable to determine the model parameter count from Hugging Face metadata "
        f"or the model name: {model_name}"
    )


def _get_weight_bits(info: Any) -> int:
    safetensors = getattr(info, "safetensors", None)
    if safetensors is not None:
        parameters = getattr(safetensors, "parameters", None)
        if parameters:
            for dtype in parameters:
                m = re.search(r"\d+", dtype)
                if m is not None:
                    return int(m.group())

    for attr_name in ("config", "cardData", "transformersInfo"):
        metadata = getattr(info, attr_name, None)
        if not isinstance(metadata, dict):
            continue
        for key in ("torch_dtype", "dtype"):
            dtype = metadata.get(key)
            if not isinstance(dtype, str):
                continue
            normalized = dtype.lower()
            if "4" in normalized and ("int" in normalized or "fp4" in normalized):
                return 4
            if "8" in normalized and ("int" in normalized or "fp8" in normalized):
                return 8
            if any(token in normalized for token in ("bfloat16", "float16", "fp16")):
                return 16
            if "float32" in normalized or normalized == "fp32":
                return 32

    # Most public foundation checkpoints default to bf16/fp16 weights.
    return 16


def get_tensor_parallel_recommendation(model_name: str):
    api = HfApi()
    info = api.model_info(model_name)
    num_parameters = _get_total_parameter_count(info, model_name)
    num_bits_in_dtype = _get_weight_bits(info)
    total_bits = num_parameters * num_bits_in_dtype

    total_model_weights_gb = total_bits / 8 / (1024**3)
    print(f"Model weights total gb: {total_model_weights_gb}")

    per_gpu_memory = determine_per_gpu_memory()
    # 0.9 is the default gpu usage for vllm
    intermediate_buffer = 5
    minimum_kv_cache_buffer = 5
    usable_per_gpu_memory = (
        per_gpu_memory * 0.9 - intermediate_buffer - minimum_kv_cache_buffer
    )
    print(f"Usable gpu memory for model weights per gpu: {usable_per_gpu_memory}")
    initial_tp = math.ceil(total_model_weights_gb / usable_per_gpu_memory)
    # round up to a power of 2
    return 2 ** math.ceil(math.log2(initial_tp))


def get_prefix_cache_token_size(model_name: str, tp: int):
    cmd = [
        "python",
        "-c",
        f"from vllm import LLM; "
        f"LLM(model='{model_name}', tensor_parallel_size={tp}, load_format='dummy')",
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # look for "GPU KV cache size:"
    # Example:
    # (EngineCore_0 pid=3166091) INFO 09-07 20:45:57
    # [kv_cache_utils.py:849] GPU KV cache size: 420,928 tokens
    # watch out if vllm ever changes their output logs in the future
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens\b", result.stdout, re.I)
    assert m is not None, "No GPU KV cache size found"
    tokens_in_prefix_cache = int(m.group(1).replace(",", ""))
    m = re.search(
        r"Available KV cache memory:\s*([\d,.]+)\s*GiB\b", result.stdout, re.I
    )
    assert m is not None, "No Available KV cache memory found"
    per_gpu_kv_cache_gb = float(m.group(1))
    return per_gpu_kv_cache_gb, tokens_in_prefix_cache


def print_vllm_deployment_string(model_name: str, tp: int):
    print("\n\n1. vLLM Deployment: \n-----------------\n")
    print(
        f"PYTHONHASHSEED=0 \\\n"
        f"vllm serve {model_name} \\\n"
        f"--tensor-parallel-size {tp} \\\n"
        f"--load-format dummy"
    )


def print_lmcache_deployment_string(
    model_name: str, tp: int, cpu_offload_GiB_per_gpu: float
):
    print("\n\n2. LMCache Deployment: \n--------------------\n")
    print(
        f"PYTHONHASHSEED=0 \\\n"
        f"LMCACHE_MAX_LOCAL_CPU_SIZE={int(cpu_offload_GiB_per_gpu)} \\\n"
        f"vllm serve {model_name} \\\n"
        f"--tensor-parallel-size {tp} \\\n"
        f"--load-format dummy \\\n"
        f"--kv-transfer-config \\\n"
        f'\'{{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}}\''
    )


def print_long_doc_qa_workload_string(model_name: str, tokens_in_offload_cache: int):
    document_length = 10000
    output_length = 100
    num_documents = int(tokens_in_offload_cache / (document_length + output_length)) - 2
    print(
        "\n\n"
        "3. Multi-Round QA Workload Generation: \n"
        "----------------------------------------\n"
    )
    print(
        f"python benchmarks/long_doc_qa/long_doc_qa.py \\\n"
        f"--model {model_name} \\\n"
        f"--num-documents {num_documents} \\\n"
        f"--document-length {document_length} \\\n"
        f"--output-len {output_length} \\\n"
        f"--repeat-count 1 \\\n"
        f"--repeat-mode tile \\\n"
        f"--max-inflight-requests 4"
    )


def get_cpu_offload_GiB_per_gpu(
    per_gpu_kv_cache_GiB: float, GiB_1K_tokens_per_gpu: float, tp: int
):
    vm = psutil.virtual_memory()
    available_pinnable_cpu_size_GiB = vm.available / 1024**3 / tp
    # `import resource` should be added at the top of the file.
    # The `resource` module is not available on Windows.
    memlock_limit_bytes, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if memlock_limit_bytes != resource.RLIM_INFINITY:
        print(f"OS restricts pinnable CPU size to {memlock_limit_bytes} bytes")
        memlock_GiB = memlock_limit_bytes / (1024**3) / tp
        available_pinnable_cpu_size_GiB = min(
            available_pinnable_cpu_size_GiB, memlock_GiB
        )
    else:
        print("You have unlimited pinnable CPU size")
    # try to allocate space for 120000 additional tokens
    DESIRED_ADDITIONAL_TOKENS_IN_OFFLOAD = 120_000
    desired_offload_GiB = (
        DESIRED_ADDITIONAL_TOKENS_IN_OFFLOAD / 1000 * GiB_1K_tokens_per_gpu
        + per_gpu_kv_cache_GiB
    )
    offload_GiB_per_gpu = min(desired_offload_GiB, available_pinnable_cpu_size_GiB)
    return offload_GiB_per_gpu


def main(model_name: str):
    tp = get_tensor_parallel_recommendation(model_name)
    print(f"Tensor Parallel Recommendation: {tp}")
    if torch.cuda.device_count() < tp:
        print(
            f"Warning: You have {torch.cuda.device_count()} GPUs, "
            f"but {model_name} requires {tp} tensor parallelism to run on your hardware"
        )
        return
    print("This will take a while...")
    per_gpu_kv_cache_GiB, tokens_in_prefix_cache = get_prefix_cache_token_size(
        model_name, tp
    )
    print(f"Tokens in prefix cache: {tokens_in_prefix_cache}")
    print(f"Per GPU KV cache GiB: {per_gpu_kv_cache_GiB}")
    GiB_1K_tokens_per_gpu = per_gpu_kv_cache_GiB / tokens_in_prefix_cache * 1000
    print(f"GiB / 1K tokens per gpu: {GiB_1K_tokens_per_gpu}")
    cpu_offload_GiB_per_gpu = get_cpu_offload_GiB_per_gpu(
        per_gpu_kv_cache_GiB, GiB_1K_tokens_per_gpu, tp
    )
    if per_gpu_kv_cache_GiB >= cpu_offload_GiB_per_gpu:
        print(
            "Warning: Your system does not have enough available pinnable CPU RAM "
            "to make use of KV Cache CPU offloading"
        )
        return
    tokens_in_offload_cache = int(
        cpu_offload_GiB_per_gpu * (1 / GiB_1K_tokens_per_gpu) * 1000
    )
    print(f"Total tokens storable: {tokens_in_offload_cache}")

    print_vllm_deployment_string(model_name, tp)
    print_lmcache_deployment_string(model_name, tp, cpu_offload_GiB_per_gpu)
    print_long_doc_qa_workload_string(model_name, tokens_in_offload_cache)


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.model)
