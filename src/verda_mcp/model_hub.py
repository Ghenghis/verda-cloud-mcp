"""
Model Hub - HuggingFace/Ollama/LM Studio integration.
MEGA-TOOL bundling 15 functions into 1 tool.
"""

from enum import Enum
from dataclasses import dataclass, field


class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"


@dataclass
class ModelInfo:
    name: str
    source: ModelSource
    size_gb: float
    parameters: str
    license: str
    vram_required: int
    gpu_rec: str
    requirements: list[str] = field(default_factory=list)


# Popular HuggingFace models
HF_MODELS = {
    "llama3-8b": ModelInfo("meta-llama/Meta-Llama-3-8B", ModelSource.HUGGINGFACE, 16, "8B", "Llama 3", 24, "A6000", ["transformers", "torch"]),
    "llama3-70b": ModelInfo("meta-llama/Meta-Llama-3-70B", ModelSource.HUGGINGFACE, 140, "70B", "Llama 3", 140, "H200", ["transformers", "torch", "accelerate"]),
    "mistral-7b": ModelInfo("mistralai/Mistral-7B-v0.3", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 16, "A6000", ["transformers", "torch"]),
    "mixtral-8x7b": ModelInfo("mistralai/Mixtral-8x7B-v0.1", ModelSource.HUGGINGFACE, 93, "46.7B MoE", "Apache 2.0", 96, "RTX PRO 6000", ["transformers", "torch"]),
    "qwen2-7b": ModelInfo("Qwen/Qwen2-7B-Instruct", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 16, "A6000", ["transformers", "torch"]),
    "qwen2-72b": ModelInfo("Qwen/Qwen2-72B-Instruct", ModelSource.HUGGINGFACE, 144, "72B", "Qwen", 144, "H200", ["transformers", "torch"]),
    "phi3-mini": ModelInfo("microsoft/Phi-3-mini-4k-instruct", ModelSource.HUGGINGFACE, 7.6, "3.8B", "MIT", 8, "A6000", ["transformers", "torch"]),
    "gemma2-9b": ModelInfo("google/gemma-2-9b-it", ModelSource.HUGGINGFACE, 18, "9B", "Gemma", 24, "A6000", ["transformers", "torch"]),
    "codellama-34b": ModelInfo("codellama/CodeLlama-34b-Instruct-hf", ModelSource.HUGGINGFACE, 68, "34B", "Llama 2", 80, "H100", ["transformers", "torch"]),
    "sdxl-base": ModelInfo("stabilityai/stable-diffusion-xl-base-1.0", ModelSource.HUGGINGFACE, 6.9, "3.5B", "RAIL++-M", 16, "A6000", ["diffusers", "torch"]),
    "whisper-large-v3": ModelInfo("openai/whisper-large-v3", ModelSource.HUGGINGFACE, 3.1, "1.5B", "MIT", 8, "A6000", ["transformers", "torch"]),
}

# Ollama models
OLLAMA_MODELS = {
    "llama3:8b": {"params": "8B", "size_gb": 4.7, "vram": 8},
    "llama3:70b": {"params": "70B", "size_gb": 40, "vram": 48},
    "mistral:7b": {"params": "7B", "size_gb": 4.1, "vram": 8},
    "mixtral:8x7b": {"params": "46.7B", "size_gb": 26, "vram": 32},
    "phi3:mini": {"params": "3.8B", "size_gb": 2.3, "vram": 4},
    "codellama:34b": {"params": "34B", "size_gb": 19, "vram": 24},
    "gemma2:9b": {"params": "9B", "size_gb": 5.4, "vram": 12},
}

# LM Studio models (GGUF)
LM_STUDIO_MODELS = {
    "llama3-8b-gguf": {"name": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", "quant": "Q4_K_M", "size_gb": 4.9, "vram": 8},
    "llama3-70b-gguf": {"name": "lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF", "quant": "Q4_K_M", "size_gb": 40, "vram": 48},
    "mistral-7b-gguf": {"name": "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF", "quant": "Q4_K_M", "size_gb": 4.4, "vram": 8},
    "phi3-mini-gguf": {"name": "lmstudio-community/Phi-3-mini-4k-instruct-GGUF", "quant": "Q4_K_M", "size_gb": 2.4, "vram": 4},
    "codellama-34b-gguf": {"name": "lmstudio-community/CodeLlama-34B-Instruct-GGUF", "quant": "Q4_K_M", "size_gb": 20, "vram": 24},
}


def model_hub(action: str = "list", model: str = "", source: str = "all") -> str:
    """
    MEGA-TOOL: Model Hub (15 functions).

    Actions: list, search, info, download_script, gpu_for_model, vram_calc,
    compare, huggingface, ollama, lm_studio, quantization, requirements,
    finetune_script, inference_script, convert_gguf
    """
    if action == "list":
        return _list_all(source)
    elif action == "search":
        return _search(model)
    elif action == "info":
        return _info(model)
    elif action == "download_script":
        return _download_script(model)
    elif action == "gpu_for_model":
        return _gpu_rec(model)
    elif action == "huggingface":
        return _list_hf()
    elif action == "ollama":
        return _list_ollama()
    elif action == "lm_studio":
        return _list_lmstudio()
    elif action == "quantization":
        return _quant_guide()
    elif action == "finetune_script":
        return _finetune(model)
    elif action == "inference_script":
        return _inference(model)
    else:
        return f"Actions: list, search, info, download_script, gpu_for_model, huggingface, ollama, lm_studio, quantization, finetune_script, inference_script"


def _list_all(source: str) -> str:
    lines = ["📦 MODEL HUB", "=" * 60]
    if source in ["all", "huggingface"]:
        lines.append("\n🤗 HUGGINGFACE")
        for k, m in HF_MODELS.items():
            lines.append(f"  {k:18} | {m.parameters:8} | {m.size_gb:5.1f}GB | {m.gpu_rec}")
    if source in ["all", "ollama"]:
        lines.append("\n🦙 OLLAMA")
        for k, m in OLLAMA_MODELS.items():
            lines.append(f"  {k:18} | {m['params']:8} | {m['size_gb']:5.1f}GB")
    if source in ["all", "lm_studio"]:
        lines.append("\n🖥️ LM STUDIO")
        for k, m in LM_STUDIO_MODELS.items():
            lines.append(f"  {k:18} | {m['quant']:8} | {m['size_gb']:5.1f}GB")
    return "\n".join(lines)


def _search(term: str) -> str:
    results = []
    for k in HF_MODELS:
        if term.lower() in k.lower():
            results.append(f"🤗 {k}")
    for k in OLLAMA_MODELS:
        if term.lower() in k.lower():
            results.append(f"🦙 {k}")
    for k in LM_STUDIO_MODELS:
        if term.lower() in k.lower():
            results.append(f"🖥️ {k}")
    return "\n".join(results) if results else f"No models matching '{term}'"


def _info(model: str) -> str:
    if model in HF_MODELS:
        m = HF_MODELS[model]
        return f"📦 {model}\nName: {m.name}\nParams: {m.parameters}\nSize: {m.size_gb}GB\nVRAM: {m.vram_required}GB\nGPU: {m.gpu_rec}"
    elif model in OLLAMA_MODELS:
        m = OLLAMA_MODELS[model]
        return f"📦 {model}\nParams: {m['params']}\nSize: {m['size_gb']}GB\nVRAM: {m['vram']}GB\nInstall: ollama pull {model}"
    elif model in LM_STUDIO_MODELS:
        m = LM_STUDIO_MODELS[model]
        return f"📦 {model}\nName: {m['name']}\nQuant: {m['quant']}\nSize: {m['size_gb']}GB"
    return f"Model '{model}' not found"


def _download_script(model: str) -> str:
    if model not in HF_MODELS:
        return "Model not in HuggingFace catalog"
    m = HF_MODELS[model]
    return f"pip install {' '.join(m.requirements)}\npython -c \"from transformers import AutoModel; AutoModel.from_pretrained('{m.name}')\""


def _gpu_rec(model: str) -> str:
    vram = 0
    if model in HF_MODELS:
        vram = HF_MODELS[model].vram_required
    elif model in OLLAMA_MODELS:
        vram = OLLAMA_MODELS[model]["vram"]
    elif model in LM_STUDIO_MODELS:
        vram = LM_STUDIO_MODELS[model]["vram"]
    if vram <= 16:
        return "✅ A6000 ($0.12/hr SPOT)"
    elif vram <= 48:
        return "✅ L40S ($0.23/hr SPOT)"
    elif vram <= 80:
        return "✅ H100 ($0.57/hr SPOT)"
    elif vram <= 141:
        return "✅ H200 ($0.75/hr SPOT)"
    return "✅ B300 ($1.24/hr SPOT)"


def _list_hf() -> str:
    return "\n".join([f"{k}: {m.parameters}" for k, m in HF_MODELS.items()])


def _list_ollama() -> str:
    return "\n".join([f"ollama pull {k}" for k in OLLAMA_MODELS])


def _list_lmstudio() -> str:
    return "\n".join([f"{k}: {m['name']}" for k, m in LM_STUDIO_MODELS.items()])


def _quant_guide() -> str:
    return "Q4_K_M: Best balance (45% size, good quality)\nQ5_K_M: Better quality (55% size)\nQ8_0: Near lossless (100% size)"


def _finetune(model: str) -> str:
    if model not in HF_MODELS:
        return "Use HuggingFace model for fine-tuning"
    m = HF_MODELS[model]
    return f"# Fine-tune {model}\nfrom peft import LoraConfig\nmodel = AutoModel.from_pretrained('{m.name}')"


def _inference(model: str) -> str:
    if model in OLLAMA_MODELS:
        return f"ollama run {model}"
    elif model in HF_MODELS:
        return f"from transformers import pipeline\npipe = pipeline('text-generation', model='{HF_MODELS[model].name}')"
    return "Model not found"
