"""Serve a model with vLLM on Modal, for testing WrenCode's openai-compatible backend.

Start it (``modal serve`` prints the endpoint URL and stops the app on Ctrl+C):

    pip install modal && modal setup
    export VLLM_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    modal serve examples/modal_vllm.py
    MODEL=Qwen/Qwen3-8B modal serve examples/modal_vllm.py   # a different model

Then point WrenCode at it:

    BACKEND=openai-compatible \\
    OPENAI_COMPATIBLE_BASE_URL=https://<workspace>--wrencode-vllm-test-serve-dev.modal.run/v1 \\
    OPENAI_COMPATIBLE_API_KEY=$VLLM_API_KEY \\
    wrencode -p "..."

GPU=L4 (the default, 24 GB) fits 7-8B models; GPU=T4 (16 GB) defaults to a 3B
model in fp16. GPUs need a payment method on the Modal account. Weights are
cached in Modal volumes, so restarts skip the download. The first request after
an idle period cold-starts the container, which can take a few minutes.
"""

import os
import subprocess

import modal

GPU = os.environ.get("GPU", "L4")
MODEL = os.environ.get(
    "MODEL", "Qwen/Qwen2.5-3B-Instruct" if GPU == "T4" else "Qwen/Qwen2.5-7B-Instruct"
)
SERVED_NAME = MODEL.rsplit("/", 1)[-1].lower()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm")
    # FlashInfer's sampler JIT-compiles with nvcc, which the slim image lacks.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("wrencode-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("wrencode-vllm-cache", create_if_missing=True)

app = modal.App("wrencode-vllm-test")


def vllm_command() -> list[str]:
    """Return the ``vllm serve`` command line, with tool calling enabled."""
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",  # reached only through Modal's web endpoint
        "--port",
        "8000",
        "--api-key",
        os.environ["VLLM_API_KEY"],
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--max-model-len",
        # 32k of KV cache doesn't fit next to an 8B model on a 24 GB L4.
        "16384" if GPU == "T4" or "8B" in MODEL else "32768",
    ]
    if GPU == "T4":
        cmd += ["--dtype", "half"]  # T4 has no bfloat16
    if "Qwen3" in MODEL:
        cmd += ["--reasoning-parser", "qwen3"]  # keep <think> out of the answer
    return cmd


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    # The container re-imports this file without the local environment, so pass
    # the choices made here (key, GPU, model) through explicitly.
    secrets=[
        modal.Secret.from_dict(
            {
                "VLLM_API_KEY": os.environ.get("VLLM_API_KEY", ""),
                "GPU": GPU,
                "MODEL": MODEL,
            }
        )
    ],
    timeout=60 * 60,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve() -> None:
    """Start vLLM in the background; Modal routes the web endpoint to port 8000."""
    if not os.environ.get("VLLM_API_KEY"):
        raise RuntimeError(
            "Set VLLM_API_KEY before `modal serve` so the endpoint isn't open."
        )
    subprocess.Popen(vllm_command())  # fixed argv, no shell
