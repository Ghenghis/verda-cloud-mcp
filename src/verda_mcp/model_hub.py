"""
Model Hub - HuggingFace/Ollama/LM Studio integration.
MEGA-TOOL bundling 20 functions into 1 tool.

v2.5.0: Expanded to 50+ models with QLoRA/LoRA configs.
"""

from dataclasses import dataclass, field
from enum import Enum


class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"


class ModelCategory(Enum):
    LLM = "llm"
    CODE = "code"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


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
    category: ModelCategory = ModelCategory.LLM
    context_length: int = 4096
    release_date: str = "2024"


@dataclass
class LoRAConfig:
    """LoRA/QLoRA configuration for fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    use_qlora: bool = False
    bits: int = 4  # 4 or 8 for QLoRA
    double_quant: bool = True
    quant_type: str = "nf4"  # nf4 or fp4


# LoRA presets based on Sebastian Raschka's research
LORA_PRESETS = {
    "minimal": LoRAConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
    "standard": LoRAConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]),
    "extended": LoRAConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    "full": LoRAConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]),
    "maximum": LoRAConfig(r=256, lora_alpha=512, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    "qlora_4bit": LoRAConfig(r=64, lora_alpha=16, use_qlora=True, bits=4, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]),
    "qlora_8bit": LoRAConfig(r=32, lora_alpha=64, use_qlora=True, bits=8, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]),
}


# Popular HuggingFace models - EXPANDED v2.5.0
HF_MODELS = {
    # === LLaMA Family ===
    "llama3-8b": ModelInfo("meta-llama/Meta-Llama-3-8B-Instruct", ModelSource.HUGGINGFACE, 16, "8B", "Llama 3", 24, "A6000", ["transformers", "torch"], ModelCategory.LLM, 8192, "2024"),
    "llama3-70b": ModelInfo("meta-llama/Meta-Llama-3-70B-Instruct", ModelSource.HUGGINGFACE, 140, "70B", "Llama 3", 140, "H200", ["transformers", "torch", "accelerate"], ModelCategory.LLM, 8192, "2024"),
    "llama3.1-8b": ModelInfo("meta-llama/Llama-3.1-8B-Instruct", ModelSource.HUGGINGFACE, 16, "8B", "Llama 3.1", 24, "A6000", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),
    "llama3.1-70b": ModelInfo("meta-llama/Llama-3.1-70B-Instruct", ModelSource.HUGGINGFACE, 140, "70B", "Llama 3.1", 140, "H200", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),
    "llama3.1-405b": ModelInfo("meta-llama/Llama-3.1-405B-Instruct", ModelSource.HUGGINGFACE, 810, "405B", "Llama 3.1", 800, "8x B300", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),
    "llama3.2-1b": ModelInfo("meta-llama/Llama-3.2-1B-Instruct", ModelSource.HUGGINGFACE, 2.4, "1B", "Llama 3.2", 4, "A6000", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),
    "llama3.2-3b": ModelInfo("meta-llama/Llama-3.2-3B-Instruct", ModelSource.HUGGINGFACE, 6, "3B", "Llama 3.2", 8, "A6000", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),

    # === Mistral Family ===
    "mistral-7b": ModelInfo("mistralai/Mistral-7B-Instruct-v0.3", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 16, "A6000", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "mixtral-8x7b": ModelInfo("mistralai/Mixtral-8x7B-Instruct-v0.1", ModelSource.HUGGINGFACE, 93, "46.7B MoE", "Apache 2.0", 96, "RTX PRO 6000", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "mixtral-8x22b": ModelInfo("mistralai/Mixtral-8x22B-Instruct-v0.1", ModelSource.HUGGINGFACE, 280, "141B MoE", "Apache 2.0", 280, "4x B300", ["transformers", "torch"], ModelCategory.LLM, 65536, "2024"),
    "mistral-nemo-12b": ModelInfo("mistralai/Mistral-Nemo-Instruct-2407", ModelSource.HUGGINGFACE, 24, "12B", "Apache 2.0", 32, "L40S", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),

    # === Qwen Family ===
    "qwen2-0.5b": ModelInfo("Qwen/Qwen2-0.5B-Instruct", ModelSource.HUGGINGFACE, 1, "0.5B", "Apache 2.0", 2, "A6000", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "qwen2-1.5b": ModelInfo("Qwen/Qwen2-1.5B-Instruct", ModelSource.HUGGINGFACE, 3, "1.5B", "Apache 2.0", 4, "A6000", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "qwen2-7b": ModelInfo("Qwen/Qwen2-7B-Instruct", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 16, "A6000", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "qwen2-72b": ModelInfo("Qwen/Qwen2-72B-Instruct", ModelSource.HUGGINGFACE, 144, "72B", "Qwen", 144, "H200", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "qwen2.5-7b": ModelInfo("Qwen/Qwen2.5-7B-Instruct", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 16, "A6000", ["transformers", "torch"], ModelCategory.LLM, 131072, "2024"),
    "qwen2.5-72b": ModelInfo("Qwen/Qwen2.5-72B-Instruct", ModelSource.HUGGINGFACE, 144, "72B", "Qwen", 144, "H200", ["transformers", "torch"], ModelCategory.LLM, 131072, "2024"),
    "qwen2.5-coder-7b": ModelInfo("Qwen/Qwen2.5-Coder-7B-Instruct", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 16, "A6000", ["transformers", "torch"], ModelCategory.CODE, 131072, "2024"),

    # === DeepSeek Family ===
    "deepseek-v2-lite": ModelInfo("deepseek-ai/DeepSeek-V2-Lite-Chat", ModelSource.HUGGINGFACE, 32, "16B", "DeepSeek", 40, "L40S", ["transformers", "torch"], ModelCategory.LLM, 32768, "2024"),
    "deepseek-coder-v2": ModelInfo("deepseek-ai/DeepSeek-Coder-V2-Instruct", ModelSource.HUGGINGFACE, 480, "236B MoE", "DeepSeek", 480, "8x B300", ["transformers", "torch"], ModelCategory.CODE, 128000, "2024"),
    "deepseek-v3": ModelInfo("deepseek-ai/DeepSeek-V3", ModelSource.HUGGINGFACE, 1340, "671B MoE", "DeepSeek", 1340, "Cluster", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),

    # === Microsoft Phi ===
    "phi3-mini": ModelInfo("microsoft/Phi-3-mini-4k-instruct", ModelSource.HUGGINGFACE, 7.6, "3.8B", "MIT", 8, "A6000", ["transformers", "torch"], ModelCategory.LLM, 4096, "2024"),
    "phi3-small": ModelInfo("microsoft/Phi-3-small-8k-instruct", ModelSource.HUGGINGFACE, 14, "7B", "MIT", 16, "A6000", ["transformers", "torch"], ModelCategory.LLM, 8192, "2024"),
    "phi3-medium": ModelInfo("microsoft/Phi-3-medium-4k-instruct", ModelSource.HUGGINGFACE, 28, "14B", "MIT", 32, "L40S", ["transformers", "torch"], ModelCategory.LLM, 4096, "2024"),
    "phi3.5-mini": ModelInfo("microsoft/Phi-3.5-mini-instruct", ModelSource.HUGGINGFACE, 7.6, "3.8B", "MIT", 8, "A6000", ["transformers", "torch"], ModelCategory.LLM, 128000, "2024"),

    # === Google Gemma ===
    "gemma2-2b": ModelInfo("google/gemma-2-2b-it", ModelSource.HUGGINGFACE, 5, "2B", "Gemma", 8, "A6000", ["transformers", "torch"], ModelCategory.LLM, 8192, "2024"),
    "gemma2-9b": ModelInfo("google/gemma-2-9b-it", ModelSource.HUGGINGFACE, 18, "9B", "Gemma", 24, "A6000", ["transformers", "torch"], ModelCategory.LLM, 8192, "2024"),
    "gemma2-27b": ModelInfo("google/gemma-2-27b-it", ModelSource.HUGGINGFACE, 54, "27B", "Gemma", 64, "H100", ["transformers", "torch"], ModelCategory.LLM, 8192, "2024"),

    # === Code Models ===
    "codellama-7b": ModelInfo("codellama/CodeLlama-7b-Instruct-hf", ModelSource.HUGGINGFACE, 14, "7B", "Llama 2", 16, "A6000", ["transformers", "torch"], ModelCategory.CODE, 16384, "2024"),
    "codellama-34b": ModelInfo("codellama/CodeLlama-34b-Instruct-hf", ModelSource.HUGGINGFACE, 68, "34B", "Llama 2", 80, "H100", ["transformers", "torch"], ModelCategory.CODE, 16384, "2024"),
    "starcoder2-7b": ModelInfo("bigcode/starcoder2-7b", ModelSource.HUGGINGFACE, 14, "7B", "BigCode", 16, "A6000", ["transformers", "torch"], ModelCategory.CODE, 16384, "2024"),
    "starcoder2-15b": ModelInfo("bigcode/starcoder2-15b", ModelSource.HUGGINGFACE, 30, "15B", "BigCode", 40, "L40S", ["transformers", "torch"], ModelCategory.CODE, 16384, "2024"),

    # === Image Generation ===
    "sdxl-base": ModelInfo("stabilityai/stable-diffusion-xl-base-1.0", ModelSource.HUGGINGFACE, 6.9, "3.5B", "RAIL++-M", 16, "A6000", ["diffusers", "torch"], ModelCategory.VISION, 0, "2023"),
    "sd3-medium": ModelInfo("stabilityai/stable-diffusion-3-medium", ModelSource.HUGGINGFACE, 4.5, "2B", "Stability", 12, "A6000", ["diffusers", "torch"], ModelCategory.VISION, 0, "2024"),
    "flux-dev": ModelInfo("black-forest-labs/FLUX.1-dev", ModelSource.HUGGINGFACE, 24, "12B", "FLUX", 32, "L40S", ["diffusers", "torch"], ModelCategory.VISION, 0, "2024"),
    "flux-schnell": ModelInfo("black-forest-labs/FLUX.1-schnell", ModelSource.HUGGINGFACE, 24, "12B", "Apache 2.0", 32, "L40S", ["diffusers", "torch"], ModelCategory.VISION, 0, "2024"),

    # === Audio/Speech ===
    "whisper-large-v3": ModelInfo("openai/whisper-large-v3", ModelSource.HUGGINGFACE, 3.1, "1.5B", "MIT", 8, "A6000", ["transformers", "torch"], ModelCategory.AUDIO, 0, "2024"),
    "whisper-large-v3-turbo": ModelInfo("openai/whisper-large-v3-turbo", ModelSource.HUGGINGFACE, 1.6, "809M", "MIT", 4, "A6000", ["transformers", "torch"], ModelCategory.AUDIO, 0, "2024"),

    # === Embeddings ===
    "bge-large": ModelInfo("BAAI/bge-large-en-v1.5", ModelSource.HUGGINGFACE, 1.3, "335M", "MIT", 2, "A6000", ["transformers", "torch"], ModelCategory.EMBEDDING, 512, "2024"),
    "bge-m3": ModelInfo("BAAI/bge-m3", ModelSource.HUGGINGFACE, 2.3, "568M", "MIT", 4, "A6000", ["transformers", "torch"], ModelCategory.EMBEDDING, 8192, "2024"),
    "e5-mistral": ModelInfo("intfloat/e5-mistral-7b-instruct", ModelSource.HUGGINGFACE, 14, "7B", "MIT", 16, "A6000", ["transformers", "torch"], ModelCategory.EMBEDDING, 32768, "2024"),
    "nomic-embed": ModelInfo("nomic-ai/nomic-embed-text-v1.5", ModelSource.HUGGINGFACE, 0.5, "137M", "Apache 2.0", 1, "A6000", ["transformers", "torch"], ModelCategory.EMBEDDING, 8192, "2024"),

    # === Multimodal ===
    "llava-1.6-7b": ModelInfo("llava-hf/llava-v1.6-mistral-7b-hf", ModelSource.HUGGINGFACE, 14, "7B", "Apache 2.0", 20, "A6000", ["transformers", "torch"], ModelCategory.MULTIMODAL, 4096, "2024"),
    "llava-1.6-34b": ModelInfo("llava-hf/llava-v1.6-34b-hf", ModelSource.HUGGINGFACE, 68, "34B", "Apache 2.0", 80, "H100", ["transformers", "torch"], ModelCategory.MULTIMODAL, 4096, "2024"),
    "idefics2-8b": ModelInfo("HuggingFaceM4/idefics2-8b", ModelSource.HUGGINGFACE, 16, "8B", "Apache 2.0", 24, "A6000", ["transformers", "torch"], ModelCategory.MULTIMODAL, 4096, "2024"),
}

# Ollama models - EXPANDED v2.5.0
OLLAMA_MODELS = {
    # LLaMA Family
    "llama3:8b": {"params": "8B", "size_gb": 4.7, "vram": 8, "category": "llm"},
    "llama3:70b": {"params": "70B", "size_gb": 40, "vram": 48, "category": "llm"},
    "llama3.1:8b": {"params": "8B", "size_gb": 4.7, "vram": 8, "category": "llm"},
    "llama3.1:70b": {"params": "70B", "size_gb": 40, "vram": 48, "category": "llm"},
    "llama3.2:1b": {"params": "1B", "size_gb": 1.3, "vram": 2, "category": "llm"},
    "llama3.2:3b": {"params": "3B", "size_gb": 2.0, "vram": 4, "category": "llm"},
    # Mistral
    "mistral:7b": {"params": "7B", "size_gb": 4.1, "vram": 8, "category": "llm"},
    "mixtral:8x7b": {"params": "46.7B", "size_gb": 26, "vram": 32, "category": "llm"},
    "mistral-nemo:12b": {"params": "12B", "size_gb": 7.1, "vram": 12, "category": "llm"},
    # Qwen
    "qwen2:7b": {"params": "7B", "size_gb": 4.4, "vram": 8, "category": "llm"},
    "qwen2:72b": {"params": "72B", "size_gb": 41, "vram": 48, "category": "llm"},
    "qwen2.5:7b": {"params": "7B", "size_gb": 4.7, "vram": 8, "category": "llm"},
    "qwen2.5-coder:7b": {"params": "7B", "size_gb": 4.7, "vram": 8, "category": "code"},
    # Phi
    "phi3:mini": {"params": "3.8B", "size_gb": 2.3, "vram": 4, "category": "llm"},
    "phi3:medium": {"params": "14B", "size_gb": 7.9, "vram": 12, "category": "llm"},
    # Gemma
    "gemma2:2b": {"params": "2B", "size_gb": 1.6, "vram": 4, "category": "llm"},
    "gemma2:9b": {"params": "9B", "size_gb": 5.4, "vram": 12, "category": "llm"},
    "gemma2:27b": {"params": "27B", "size_gb": 16, "vram": 24, "category": "llm"},
    # Code
    "codellama:7b": {"params": "7B", "size_gb": 3.8, "vram": 8, "category": "code"},
    "codellama:34b": {"params": "34B", "size_gb": 19, "vram": 24, "category": "code"},
    "deepseek-coder:6.7b": {"params": "6.7B", "size_gb": 3.8, "vram": 8, "category": "code"},
    "starcoder2:7b": {"params": "7B", "size_gb": 4.0, "vram": 8, "category": "code"},
    # Embeddings
    "nomic-embed-text": {"params": "137M", "size_gb": 0.27, "vram": 1, "category": "embedding"},
    "mxbai-embed-large": {"params": "335M", "size_gb": 0.67, "vram": 2, "category": "embedding"},
    # Multimodal
    "llava:7b": {"params": "7B", "size_gb": 4.5, "vram": 10, "category": "multimodal"},
    "llava:34b": {"params": "34B", "size_gb": 20, "vram": 28, "category": "multimodal"},
}

# LM Studio models (GGUF)
LM_STUDIO_MODELS = {
    "llama3-8b-gguf": {"name": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF", "quant": "Q4_K_M", "size_gb": 4.9, "vram": 8},
    "llama3-70b-gguf": {"name": "lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF", "quant": "Q4_K_M", "size_gb": 40, "vram": 48},
    "mistral-7b-gguf": {"name": "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF", "quant": "Q4_K_M", "size_gb": 4.4, "vram": 8},
    "phi3-mini-gguf": {"name": "lmstudio-community/Phi-3-mini-4k-instruct-GGUF", "quant": "Q4_K_M", "size_gb": 2.4, "vram": 4},
    "codellama-34b-gguf": {"name": "lmstudio-community/CodeLlama-34B-Instruct-GGUF", "quant": "Q4_K_M", "size_gb": 20, "vram": 24},
}


def model_hub(action: str = "list", model: str = "", source: str = "all", preset: str = "standard") -> str:
    """
    MEGA-TOOL: Model Hub (20 functions).

    v2.5.0: Expanded to 50+ models with QLoRA/LoRA configs.

    Actions: list, search, info, download_script, gpu_for_model, vram_calc,
    compare, huggingface, ollama, lm_studio, quantization, requirements,
    finetune_script, inference_script, convert_gguf, lora_config, qlora_config,
    lora_presets, category, stats
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
        return _finetune(model, preset)
    elif action == "inference_script":
        return _inference(model)
    elif action == "lora_config":
        return _lora_config(preset)
    elif action == "qlora_config":
        return _qlora_config(model)
    elif action == "lora_presets":
        return _list_lora_presets()
    elif action == "category":
        return _list_by_category(source)
    elif action == "stats":
        return _model_stats()
    else:
        return """Actions: list, search, info, download_script, gpu_for_model,
huggingface, ollama, lm_studio, quantization, finetune_script,
inference_script, lora_config, qlora_config, lora_presets, category, stats"""


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


def _finetune(model: str, preset: str = "standard") -> str:
    if model not in HF_MODELS:
        return "Use HuggingFace model for fine-tuning"
    m = HF_MODELS[model]
    lora = LORA_PRESETS.get(preset, LORA_PRESETS["standard"])

    script = f'''# Fine-tune {model} with {preset.upper()} LoRA preset
# GPU Recommendation: {m.gpu_rec} | VRAM: {m.vram_required}GB

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{m.name}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("{m.name}")
tokenizer.pad_token = tokenizer.eos_token

# LoRA config ({preset})
lora_config = LoraConfig(
    r={lora.r},
    lora_alpha={lora.lora_alpha},
    lora_dropout={lora.lora_dropout},
    target_modules={lora.target_modules},
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./output/{model}",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("./output/{model}-lora")
'''
    return script


def _inference(model: str) -> str:
    if model in OLLAMA_MODELS:
        return f"ollama run {model}"
    elif model in HF_MODELS:
        return f"from transformers import pipeline\npipe = pipeline('text-generation', model='{HF_MODELS[model].name}')"
    return "Model not found"


def _lora_config(preset: str) -> str:
    """Get LoRA configuration for a preset."""
    if preset not in LORA_PRESETS:
        return f"Unknown preset. Available: {', '.join(LORA_PRESETS.keys())}"

    lora = LORA_PRESETS[preset]
    return f'''# LoRA Config: {preset.upper()}
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r={lora.r},
    lora_alpha={lora.lora_alpha},
    lora_dropout={lora.lora_dropout},
    target_modules={lora.target_modules},
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

# Trainable params: ~{lora.r * len(lora.target_modules) * 2}M for 7B model
# Memory overhead: ~{lora.r * 8}MB per layer
'''


def _qlora_config(model: str) -> str:
    """Get QLoRA 4-bit configuration for memory-efficient training."""
    m = HF_MODELS.get(model)
    model_name = m.name if m else "your-model-name"

    return f'''# QLoRA 4-bit Configuration
# Reduces VRAM by ~75% compared to full fine-tuning
# Based on: https://arxiv.org/abs/2305.14314

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# QLoRA config (r=64 recommended for 4-bit)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Install: pip install bitsandbytes>=0.41.0
'''


def _list_lora_presets() -> str:
    """List all available LoRA presets."""
    lines = ["🔧 LoRA PRESETS (Based on Sebastian Raschka's Research)", "=" * 60]
    for name, config in LORA_PRESETS.items():
        qlora_tag = " [QLoRA]" if config.use_qlora else ""
        lines.append(f"\n{name.upper()}{qlora_tag}")
        lines.append(f"  r={config.r}, alpha={config.lora_alpha}")
        lines.append(f"  Target: {', '.join(config.target_modules)}")
        if config.use_qlora:
            lines.append(f"  Bits: {config.bits}-bit, Quant: {config.quant_type}")

    lines.append("\n💡 RECOMMENDATIONS:")
    lines.append("  - minimal: Quick experiments, low VRAM")
    lines.append("  - standard: Balanced quality/speed")
    lines.append("  - extended: Better quality, more layers")
    lines.append("  - maximum: Best quality (r=256, alpha=512)")
    lines.append("  - qlora_4bit: 75% VRAM reduction")
    return "\n".join(lines)


def _list_by_category(category: str) -> str:
    """List models by category."""
    lines = [f"📦 MODELS BY CATEGORY: {category.upper()}", "=" * 60]

    cat_map = {
        "llm": ModelCategory.LLM,
        "code": ModelCategory.CODE,
        "vision": ModelCategory.VISION,
        "audio": ModelCategory.AUDIO,
        "embedding": ModelCategory.EMBEDDING,
        "multimodal": ModelCategory.MULTIMODAL,
    }

    if category == "all":
        for cat in ModelCategory:
            lines.append(f"\n{cat.value.upper()}")
            for k, m in HF_MODELS.items():
                if m.category == cat:
                    lines.append(f"  {k:20} | {m.parameters:8} | {m.vram_required:3}GB VRAM")
    elif category in cat_map:
        target = cat_map[category]
        for k, m in HF_MODELS.items():
            if m.category == target:
                lines.append(f"  {k:20} | {m.parameters:8} | {m.vram_required:3}GB VRAM | {m.context_length:,} ctx")
    else:
        return "Categories: all, llm, code, vision, audio, embedding, multimodal"

    return "\n".join(lines)


def _model_stats() -> str:
    """Show model hub statistics."""
    hf_count = len(HF_MODELS)
    ollama_count = len(OLLAMA_MODELS)
    lmstudio_count = len(LM_STUDIO_MODELS)
    total = hf_count + ollama_count + lmstudio_count

    # Count by category
    cats = {}
    for m in HF_MODELS.values():
        cats[m.category.value] = cats.get(m.category.value, 0) + 1

    lines = [
        "📊 MODEL HUB STATISTICS",
        "=" * 60,
        f"\n🔢 TOTAL MODELS: {total}",
        f"  🤗 HuggingFace: {hf_count}",
        f"  🦙 Ollama: {ollama_count}",
        f"  🖥️ LM Studio: {lmstudio_count}",
        "\n📁 BY CATEGORY (HuggingFace):",
    ]
    for cat, count in sorted(cats.items()):
        lines.append(f"  {cat:12}: {count}")

    lines.append(f"\n🔧 LoRA PRESETS: {len(LORA_PRESETS)}")

    return "\n".join(lines)
