"""
Simple inference script: FLUX.2-Klein-4B → PiD pixel decoder → 2048² image.

This script:
  1. Loads FLUX.2-Klein-4B from a local path via diffusers
  2. Generates a latent at 512×512 using text-to-image
  3. Loads PiD decoder and decodes the latent into 2048² pixels

Prerequisites:
  - diffusers >= 0.37.0  (pip install diffusers>=0.37.0)
  - PiD flux2 checkpoint  (bash download_checkpoints.sh flux2)

Usage:
  cd auto_remaster/sandbox/PiD
  PYTHONPATH=. python run_flux2_klein_pid.py \
      --prompt "A cinematic still of a fox in autumn leaves" \
      --output_dir ./results/klein_pid
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("run_flux2_klein_pid")

# ─── Paths ───────────────────────────────────────────────────────────────────
FLUX2_KLEIN_PATH = "/code/models/FLUX.2-klein-4B"
PID_EXPERIMENT = "PiD_res2k_sr4x_official_flux2_distill_4step"
PID_CHECKPOINT = "/code/auto_remaster/sandbox/PiD/checkpoints"
PID_CONFIG = "pid/_src/configs/pid/config.py"


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2-Klein + PiD inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--flux2_path", type=str, default=FLUX2_KLEIN_PATH, help="Path to FLUX.2-Klein-4B")
    parser.add_argument("--pid_checkpoint", type=str, default=PID_CHECKPOINT, help="PiD checkpoint path")
    parser.add_argument("--resolution", type=int, default=512, help="LDM generation resolution")
    parser.add_argument("--ldm_steps", type=int, default=28, help="FLUX.2 Klein denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="CFG scale for FLUX.2")
    parser.add_argument("--pid_steps", type=int, default=4, help="PiD denoising steps")
    parser.add_argument("--pid_cfg_scale", type=float, default=1.0, help="PiD CFG scale")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (4 = 512→2048)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results/klein_pid", help="Output directory")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offload for FLUX.2")
    parser.add_argument("--skip_pid", action="store_true", help="Only run FLUX.2 + VAE decode (no PiD)")
    return parser.parse_args()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] in [-1, 1] to PIL Image."""
    tensor = (tensor.float().clamp(-1, 1) + 1) * 127.5
    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


# ─── Step 1: Load FLUX.2 Klein pipeline ──────────────────────────────────────
def load_flux2_klein(model_path: str, cpu_offload: bool = False):
    """Load FLUX.2-Klein-4B pipeline from local disk."""
    # Dynamic import — requires diffusers >= 0.37
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        logger.error(
            "Flux2KleinPipeline not found. Install diffusers >= 0.37:\n"
            "  pip install diffusers>=0.37.0"
        )
        sys.exit(1)

    logger.info(f"Loading FLUX.2-Klein-4B from {model_path} ...")
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if cpu_offload:
        pipeline.enable_model_cpu_offload()
        logger.info("CPU offload enabled.")
    else:
        pipeline = pipeline.to("cuda")

    return pipeline


# ─── Step 2: Generate latent ─────────────────────────────────────────────────
def generate_latent(pipeline, prompt: str, height: int, width: int,
                    num_steps: int, guidance_scale: float, seed: int):
    """Run FLUX.2 pipeline and return the raw latent + sigma."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    logger.info(f"Generating latent: {prompt[:80]!r} at {height}×{width}, {num_steps} steps ...")
    output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        output_type="latent",
        generator=generator,
        max_sequence_length=512,
    )

    # Unpack FLUX.2 packed latent to (B, C, H, W)
    latent_packed = output.images  # (B, seq_len, C) packed

    try:
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
        vae_sf = pipeline.vae_scale_factor  # typically 8
        latent_h = height // (vae_sf * 2)
        latent_w = width // (vae_sf * 2)
        dummy = torch.zeros(latent_packed.shape[0], 1, latent_h, latent_w, device=latent_packed.device)
        latent_ids = Flux2Pipeline._prepare_latent_ids(dummy).to(latent_packed.device)
        latent = Flux2Pipeline._unpack_latents_with_ids(latent_packed, latent_ids)
        if not isinstance(latent, torch.Tensor):
            latent = torch.stack(latent, dim=0)
    except Exception:
        # Fallback: try Klein-specific unpacking
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as _Klein
        vae_sf = pipeline.vae_scale_factor
        latent_h = height // (vae_sf * 2)
        latent_w = width // (vae_sf * 2)
        latent = latent_packed.view(latent_packed.shape[0], latent_h, latent_w, -1)
        latent = latent.permute(0, 3, 1, 2)

    # Get final sigma from scheduler
    sigma = float(pipeline.scheduler.sigmas[-1].item())

    logger.info(f"Latent shape: {list(latent.shape)}, sigma: {sigma:.4f}")
    return latent, sigma


# ─── Step 3: VAE decode (baseline) ──────────────────────────────────────────
def vae_decode(pipeline, latent: torch.Tensor) -> torch.Tensor:
    """Standard VAE decode for baseline comparison. Returns [B, 3, H, W] in [-1, 1]."""
    # Denormalize via BatchNorm running stats
    bn = pipeline.vae.bn
    bn_mean = bn.running_mean.to(latent.device, latent.dtype).view(1, -1, 1, 1)
    bn_var = bn.running_var.to(latent.device, latent.dtype)
    bn_std = (bn_var + bn.eps).sqrt().view(1, -1, 1, 1)
    raw_latent = latent * bn_std + bn_mean

    # Unpatchify 2x2: (B, 128, h, w) → (B, 32, 2h, 2w)
    try:
        from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
        raw_latent = Flux2Pipeline._unpatchify_latents(raw_latent)
    except Exception:
        from einops import rearrange
        raw_latent = rearrange(
            raw_latent,
            "b (c pi pj) i j -> b c (i pi) (j pj)",
            pi=2, pj=2,
        )

    raw_latent = raw_latent.to(pipeline.vae.dtype)
    with torch.no_grad():
        decoded = pipeline.vae.decode(raw_latent, return_dict=False)[0]

    return decoded  # [-1, 1]


# ─── Step 4: PiD decode ─────────────────────────────────────────────────────
def load_pid_model(checkpoint_path: str, experiment: str, config_file: str):
    """Load PiD pixel decoder from checkpoint."""
    logger.info(f"Loading PiD decoder from {checkpoint_path} ...")

    from pid._src.utils.model_loader import load_model_from_checkpoint
    model, config = load_model_from_checkpoint(
        experiment_name=experiment,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        enable_fsdp=False,
        strict=False,
    )
    model.eval()
    logger.info("PiD decoder loaded.")
    return model


def pid_decode(model, latent: torch.Tensor, prompt: str,
               sigma: float, scale: int, pid_steps: int,
               pid_cfg_scale: float, seed: int) -> torch.Tensor:
    """Run PiD pixel decoder on a FLUX.2 latent. Returns [C, 1, H_out, W_out] in [-1, 1]."""
    B = latent.shape[0]
    lq_h = latent.shape[-2] * 16  # FLUX.2 VAE: 16× spatial compression
    lq_w = latent.shape[-1] * 16
    target_h = lq_h * scale
    target_w = lq_w * scale

    # PiD needs: caption, LQ_latent, degrade_sigma
    # LQ_video_or_image can be zeros placeholder (lq_condition_type="latent" for flux2)
    lq_placeholder = torch.zeros(B, 3, lq_h, lq_w, dtype=torch.bfloat16, device="cuda")

    data_batch = {
        model.config.input_caption_key: [prompt] * B,
        "LQ_video_or_image": lq_placeholder,
        "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([sigma] * B, device="cuda", dtype=torch.float32),
    }

    logger.info(f"PiD decode: latent {list(latent.shape)} → pixels ({target_h}×{target_w}), "
                f"{pid_steps} steps, cfg={pid_cfg_scale}")

    with torch.no_grad():
        samples = model.generate_samples_from_batch(
            data_batch,
            cfg_scale=pid_cfg_scale,
            num_steps=pid_steps,
            seed=seed,
            image_size=(target_h, target_w),
        )

    return samples  # [B, C, 1, H_out, W_out] in [-1, 1]


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.enable_grad(False)

    # 1. Load FLUX.2 Klein
    pipeline = load_flux2_klein(args.flux2_path, args.cpu_offload)

    # 2. Generate latent
    latent, sigma = generate_latent(
        pipeline, args.prompt,
        height=args.resolution, width=args.resolution,
        num_steps=args.ldm_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # 3. VAE decode (baseline)
    logger.info("VAE decoding (baseline) ...")
    vae_image = vae_decode(pipeline, latent)
    vae_pil = tensor_to_pil(vae_image[0])
    vae_path = os.path.join(args.output_dir, "vae_decode.jpg")
    vae_pil.save(vae_path, quality=95)
    logger.info(f"VAE baseline saved: {vae_path} ({vae_pil.size})")

    # Free FLUX.2 memory before loading PiD
    del pipeline
    torch.cuda.empty_cache()

    if args.skip_pid:
        logger.info("--skip_pid set, skipping PiD decode.")
        return

    # 4. PiD decode
    if not os.path.exists(args.pid_checkpoint):
        logger.error(
            f"PiD checkpoint not found: {args.pid_checkpoint}\n"
            f"Download it first:\n"
            f"  bash download_checkpoints.sh flux2"
        )
        sys.exit(1)

    pid_model = load_pid_model(args.pid_checkpoint, PID_EXPERIMENT, PID_CONFIG)

    samples = pid_decode(
        pid_model, latent, args.prompt,
        sigma=sigma, scale=args.scale,
        pid_steps=args.pid_steps,
        pid_cfg_scale=args.pid_cfg_scale,
        seed=args.seed,
    )

    # Save output
    pid_image = samples[0].float().cpu().clamp(-1, 1)
    if pid_image.dim() == 4:
        pid_image = pid_image.squeeze(1)  # remove T dim
    pid_pil = tensor_to_pil(pid_image)
    pid_path = os.path.join(args.output_dir, "pid_decode.jpg")
    pid_pil.save(pid_path, quality=95)
    logger.info(f"PiD output saved: {pid_path} ({pid_pil.size})")

    # 5. Save side-by-side comparison
    vae_resized = vae_pil.resize(pid_pil.size, Image.BICUBIC)
    comparison = Image.new("RGB", (pid_pil.width * 2, pid_pil.height))
    comparison.paste(vae_resized, (0, 0))
    comparison.paste(pid_pil, (pid_pil.width, 0))
    cmp_path = os.path.join(args.output_dir, "comparison.jpg")
    comparison.save(cmp_path, quality=95)
    logger.info(f"Side-by-side comparison saved: {cmp_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
