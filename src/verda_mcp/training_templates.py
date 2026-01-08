"""
Training Templates - Pre-built configs for popular models.
MEGA-TOOL bundling 10 functions into 1 tool.
"""

from dataclasses import dataclass
from enum import Enum


class Category(Enum):
    LLM = "llm_finetune"
    CODE = "code"
    IMAGE = "image_gen"
    SPEECH = "speech"
    EMBEDDING = "embedding"


@dataclass
class Template:
    name: str
    category: Category
    model: str
    gpu: str
    gpu_count: int
    batch_size: int
    lr: float
    epochs: int
    desc: str


TEMPLATES = {
    "llama3-lora": Template("LLaMA 3 LoRA", Category.LLM, "llama3-8b", "A6000", 2, 4, 2e-5, 3, "Standard LoRA fine-tuning"),
    "llama3-full": Template("LLaMA 3 Full", Category.LLM, "llama3-8b", "H100", 4, 2, 1e-5, 3, "Full fine-tuning"),
    "mistral-lora": Template("Mistral LoRA", Category.LLM, "mistral-7b", "A6000", 2, 4, 2e-5, 3, "Mistral 7B LoRA"),
    "codellama-lora": Template("CodeLLaMA LoRA", Category.CODE, "codellama-34b", "H100", 2, 2, 1e-5, 3, "Code generation"),
    "deepseek-coder": Template("DeepSeek Coder", Category.CODE, "deepseek-33b", "H100", 2, 2, 1e-5, 3, "Code assistant"),
    "sdxl-lora": Template("SDXL LoRA", Category.IMAGE, "sdxl-base", "A6000", 1, 1, 1e-6, 1000, "Image generation"),
    "flux-lora": Template("FLUX LoRA", Category.IMAGE, "flux-dev", "L40S", 1, 1, 1e-6, 500, "FLUX fine-tuning"),
    "whisper-ft": Template("Whisper Fine-tune", Category.SPEECH, "whisper-large", "A6000", 1, 8, 1e-5, 5, "Speech recognition"),
    "bge-embed": Template("BGE Embedding", Category.EMBEDDING, "bge-large", "A6000", 1, 32, 2e-5, 3, "Custom embeddings"),
    "e5-embed": Template("E5 Embedding", Category.EMBEDDING, "e5-mistral", "A6000", 2, 16, 2e-5, 3, "E5 embeddings"),
}


def training_templates(action: str = "list", template: str = "", **kwargs) -> str:
    """
    MEGA-TOOL: Training Templates (10 functions).

    Actions: list, info, generate, config, requirements, gpu_setup,
    data_format, customize, validate, estimate
    """
    if action == "list":
        lines = ["📋 TRAINING TEMPLATES", "=" * 70]
        for cat in Category:
            lines.append(f"\n{cat.value.upper()}")
            for k, t in TEMPLATES.items():
                if t.category == cat:
                    lines.append(f"  {k:18} | {t.gpu} x{t.gpu_count} | {t.desc}")
        return "\n".join(lines)

    elif action == "info":
        if template not in TEMPLATES:
            return f"Template '{template}' not found"
        t = TEMPLATES[template]
        return f"""
📋 TEMPLATE: {t.name}
═══════════════════════════════════════════════════
Category:     {t.category.value}
Model:        {t.model}
GPU:          {t.gpu} x{t.gpu_count}
Batch Size:   {t.batch_size}
Learning Rate:{t.lr}
Epochs:       {t.epochs}
Description:  {t.desc}

Generate script: training_templates(action='generate', template='{template}')
"""

    elif action == "generate":
        if template not in TEMPLATES:
            return "Template not found"
        t = TEMPLATES[template]
        if t.category == Category.LLM:
            return f'''# {t.name} Training Script
# GPU: {t.gpu} x{t.gpu_count} | Batch: {t.batch_size} | LR: {t.lr}

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("{t.model}", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{t.model}")

# LoRA config
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora)

# Training
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs={t.epochs},
    per_device_train_batch_size={t.batch_size},
    gradient_accumulation_steps=4,
    learning_rate={t.lr},
    bf16=True,
    save_steps=100,
    logging_steps=10,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
'''
        elif t.category == Category.IMAGE:
            return f'''# {t.name} Training Script
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# Add LoRA training with kohya-ss/sd-scripts
'''
        return f"# Template for {t.category.value}"

    elif action == "config":
        if template not in TEMPLATES:
            return "Template not found"
        t = TEMPLATES[template]
        return f'''{{"model": "{t.model}", "gpu": "{t.gpu}", "gpu_count": {t.gpu_count}, "batch_size": {t.batch_size}, "lr": {t.lr}, "epochs": {t.epochs}}}'''

    elif action == "requirements":
        return "pip install transformers torch accelerate peft bitsandbytes datasets wandb"

    elif action == "gpu_setup":
        return "nvidia-smi && python -c 'import torch; print(f\"GPUs: {torch.cuda.device_count()}\")'"

    elif action == "data_format":
        return '''# Expected data format (JSONL)
{"instruction": "Summarize this text", "input": "Long text here...", "output": "Summary"}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}'''

    elif action == "customize":
        return "# Modify template parameters in TrainingArguments"

    elif action == "validate":
        return "# Check GPU memory, dataset format, model loading"

    elif action == "estimate":
        if template not in TEMPLATES:
            return "Template not found"
        t = TEMPLATES[template]
        hours = t.epochs * 2
        cost = {"A6000": 0.12, "L40S": 0.23, "H100": 0.57}.get(t.gpu, 0.5) * t.gpu_count * hours
        return f"⏱️ Est. {hours}h on {t.gpu} x{t.gpu_count} SPOT = ${cost:.2f}"

    return "Actions: list, info, generate, config, requirements, gpu_setup, data_format, customize, validate, estimate"
