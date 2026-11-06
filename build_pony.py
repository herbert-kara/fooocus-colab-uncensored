import json

src = [
    "!pip install -q diffusers transformers accelerate torch torchvision gradio safetensors huggingface_hub\n",
    "\n",
    "import torch\n",
    "from diffusers import StableDiffusionXLImg2ImgPipeline\n",
    "import gradio as gr\n",
    "import os\n",
    "\n",
    "has_cuda = torch.cuda.is_available()\n",
    "device = \"cuda\" if has_cuda else \"cpu\"\n",
    "dtype = torch.float16 if has_cuda else torch.float32\n",
    "\n",
    "print(\"Loading Pony V6 XL Turbo DPO (diffusers format, low VRAM)...\")\n",
    "pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(\n",
    "    \"Bakanayatsu/ponyDiffusion-V6-XL-Turbo-DPO\",\n",
    "    torch_dtype=dtype,\n",
    "    use_safetensors=True\n",
    ")\n",
    "\n",
    "# Turbo build: 8-12 steps, lower VRAM peak than full checkpoint\n",
    "\n",
    "if hasattr(pipe_i2i, \"safety_checker\"):\n",
    "    pipe_i2i.safety_checker = None\n",
    "if hasattr(pipe_i2i, \"requires_safety_checker\"):\n",
    "    pipe_i2i.requires_safety_checker = False\n",
    "\n",
    "if has_cuda:\n",
    "    pipe_i2i.enable_sequential_cpu_offload()\n",
    "else:\n",
    "    pipe_i2i.to(\"cpu\")\n",
    "\n",
    "def edit(image, prompt, negative_prompt, strength, steps, cfg):\n",
    "    if image is None:\n",
    "        return None\n",
    "    image = image.convert(\"RGB\")\n",
    "    image.thumbnail((1024, 1024))\n",
    "    full_prompt = \"score_9, score_8_up, score_7_up, \" + prompt\n",
    "    full_neg = \"score_6, score_5, score_4, \" + negative_prompt\n",
    "    img = pipe_i2i(\n",
    "        prompt=full_prompt,\n",
    "        negative_prompt=full_neg,\n",
    "        image=image,\n",
    "        strength=float(strength),\n",
    "        num_inference_steps=int(steps),\n",
    "        guidance_scale=float(cfg)\n",
    "    ).images[0]\n",
    "    return img\n",
    "\n",
    "demo = gr.Interface(\n",
    "    fn=edit,\n",
    "    inputs=[\n",
    "        gr.Image(type=\"pil\", label=\"Source Image\"),\n",
    "        gr.Textbox(label=\"Prompt (tag format e.g. solo, expressive, ...)\", value=\"solo, masterpiece, highly detailed\"),\n",
    "        gr.Textbox(label=\"Negative Prompt\", value=\"source_low, lowres, bad anatomy\"),\n",
    "        gr.Slider(0.1, 0.6, value=0.25, label=\"Strength (0.2 - 0.3 preserves identity)\"),\n",
    "        gr.Slider(8, 30, value=12, step=1, label=\"Steps\"),\n",
    "        gr.Slider(4.0, 10.0, value=7.0, step=0.5, label=\"CFG Scale\")\n",
    "    ],\n",
    "    outputs=gr.Image(label=\"Result (Uncensored)\"),\n",
    "    title=\"Pony Diffusion V6 XL Uncensored Studio\"\n",
    ")\n",
    "demo.launch(share=True, debug=True)\n",
]

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Pony V6 XL Turbo DPO - Uncensored Studio (Low-VRAM)\n",
            "Turbo-DPO checkpoint in diffusers format (~7 GB total, verified HTTP 200, non-gated). Fits T4 with CPU offload."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src},
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 2,
}

with open("Z:/hermes/scratch/RealVisXL_Separate_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# validate JSON round-trip
json.load(open("Z:/hermes/scratch/RealVisXL_Separate_Pipeline.ipynb", encoding="utf-8"))
print("OK: JSON valid, notebook written")
