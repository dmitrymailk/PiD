#!/bin/bash
# Download PiD checkpoints from HuggingFace Hub
# Source: https://huggingface.co/nvidia/PiD
#
# This downloads only the checkpoints/ directory into the current repo.
# For FLUX.2 you need: PiD_res2k_sr4x_official_flux2_distill_4step
#
# Usage:
#   cd auto_remaster/sandbox/PiD
#   bash download_checkpoints.sh          # download all checkpoints
#   bash download_checkpoints.sh flux2    # download only flux2 checkpoint

set -e

REPO_ID="nvidia/PiD"
export HF_HOME=/code/models/huggingface
LOCAL_DIR="."

if [ "$1" == "flux2" ]; then
    echo "Downloading FLUX.2 PiD checkpoint + VAE..."
    huggingface-cli download "$REPO_ID" \
        --local-dir "$LOCAL_DIR" \
        --include "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/*" "checkpoints/flux2_ae.safetensors"
elif [ "$1" == "flux" ]; then
    echo "Downloading FLUX.1 PiD checkpoint only..."
    huggingface-cli download "$REPO_ID" \
        --local-dir "$LOCAL_DIR" \
        --include "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/*"
elif [ "$1" == "sd3" ]; then
    echo "Downloading SD3 PiD checkpoint only..."
    huggingface-cli download "$REPO_ID" \
        --local-dir "$LOCAL_DIR" \
        --include "checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/*"
elif [ "$1" == "vae" ]; then
    echo "Downloading FLUX.2 VAE only..."
    huggingface-cli download "$REPO_ID" \
        --local-dir "$LOCAL_DIR" \
        --include "checkpoints/flux2_ae.safetensors"
elif [ "$1" == "all" ] || [ -z "$1" ]; then
    echo "Downloading ALL PiD checkpoints..."
    huggingface-cli download "$REPO_ID" \
        --local-dir "$LOCAL_DIR" \
        --include "checkpoints/*"
else
    echo "Usage: bash download_checkpoints.sh [flux2|flux|sd3|vae|all]"
    echo "  flux2  - FLUX.2 decoder + VAE (~2.8GB)"
    echo "  flux   - FLUX.1 decoder only (~2.6GB)"
    echo "  sd3    - SD3 decoder only (~2.6GB)"
    echo "  vae    - FLUX.2 VAE only (~300MB)"
    echo "  all    - All decoders (~20GB+)"
    exit 1
fi

echo "Done! Checkpoints saved to ./checkpoints/"
