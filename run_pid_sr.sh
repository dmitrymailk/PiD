#!/bin/bash
# PiD super-resolution from VAE latent
# Usage: bash run_pid_sr.sh /path/to/image.jpg

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Cache HuggingFace models (Gemma-2-2B-it etc.) to a persistent location
export HF_HOME=/code/models/huggingface

INPUT_PATH=/code/checkpoints/distillation/baseline_flow_struct_noise_diff_v3/validation/step-120000/generated/sample_03000.png
PROMPT="${2:-make this image photorealistic,high quality}"

PYTHONPATH=. python run_pid_from_vae_latent.py \
    --input_path "$INPUT_PATH" \
    --prompt "$PROMPT" \
    --use_pid_vae \
    --resolution 512 \
    --pid_steps 1 \
    --scale 4 \
    --seed 42 \
    --output_dir ./results/pid_sr
