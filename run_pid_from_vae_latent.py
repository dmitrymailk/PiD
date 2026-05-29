"""
PiD as pure super-resolution: Image → FLUX.2 VAE encode → PiD decode → 4× upscaled image.

No FLUX.2 transformer is used. This script:
  1. Loads ONLY the FLUX.2 VAE from FLUX.2-Klein-4B (no 4B transformer needed)
  2. Encodes an input image to latent [1, 128, H/16, W/16]
  3. Feeds the clean latent (sigma=0) to PiD → outputs 4× resolution pixels

This works because PiD was trained on clean VAE latents (sigma=0) as one of
its conditioning modes. The latent provides structure, PiD synthesizes details.

Prerequisites:
  - diffusers >= 0.37.0
  - PiD flux2 checkpoint (bash download_checkpoints.sh flux2)

Usage:
  cd auto_remaster/sandbox/PiD

  # Single image:
  PYTHONPATH=. python run_pid_from_vae_latent.py \
      --input_path /path/to/image.jpg \
      --prompt "a high quality photograph" \
      --output_dir ./results/pid_sr

  # Use PiD's built-in VAE (no diffusers needed):
  PYTHONPATH=. python run_pid_from_vae_latent.py \
      --input_path /path/to/image.jpg \
      --prompt "a high quality photograph" \
      --use_pid_vae \
      --output_dir ./results/pid_sr
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("pid_from_vae")

# ─── Defaults ────────────────────────────────────────────────────────────────
FLUX2_KLEIN_PATH = os.environ.get(
    "FLUX2_KLEIN_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "FLUX.2-klein-4B"),
)
PID_EXPERIMENT = "PiD_res2k_sr4x_official_flux2_distill_4step"
PID_CHECKPOINT = "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth"
PID_CONFIG = "pid/_src/configs/pid/config.py"


def parse_args():
    parser = argparse.ArgumentParser(description="PiD SR from VAE latent (no LDM needed)")
    parser.add_argument("--input_path", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, default="high quality, detailed, sharp",
                        help="Text prompt describing the image (helps PiD generate better details)")
    parser.add_argument("--flux2_path", type=str, default=FLUX2_KLEIN_PATH,
                        help="Path to FLUX.2-Klein-4B (only VAE is loaded)")
    parser.add_argument("--use_pid_vae", action="store_true",
                        help="Use PiD's built-in FLUX.2 VAE instead of loading from diffusers. "
                             "No diffusers dependency needed, but requires the VAE safetensors "
                             "in the PiD checkpoint tree.")
    parser.add_argument("--pid_checkpoint", type=str, default=PID_CHECKPOINT)
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resize input to this resolution before encoding (0 = keep original)")
    parser.add_argument("--pid_steps", type=int, default=4, help="PiD denoising steps")
    parser.add_argument("--pid_cfg_scale", type=float, default=1.0, help="PiD CFG scale")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Noise level to add to latent (0.0 = clean, >0 gives PiD more creative freedom)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/pid_sr")
    return parser.parse_args()


# ─── Image I/O ───────────────────────────────────────────────────────────────

def load_image(path: str, resolution: int = 512) -> torch.Tensor:
    """Load image, center-crop to square, resize. Returns [1, 3, H, W] in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if resolution > 0:
        # Center-crop to square, then resize
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((resolution, resolution), Image.BICUBIC)
    else:
        # Keep original size, just crop to multiple of 16
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        if (new_w, new_h) != (w, h):
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))

    arr = np.asarray(img, np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] in [-1, 1] to PIL."""
    tensor = (tensor.float().clamp(-1, 1) + 1) * 127.5
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8))


# ─── Option A: Load FLUX.2 VAE from diffusers ───────────────────────────────

def encode_with_diffusers_vae(image: torch.Tensor, flux2_path: str) -> torch.Tensor:
    """Encode image using FLUX.2 VAE from diffusers (only loads VAE, not transformer)."""
    try:
        from diffusers import AutoencoderKLFlux2
    except ImportError:
        logger.error("AutoencoderKLFlux2 not found. Install diffusers >= 0.37 or use --use_pid_vae")
        sys.exit(1)

    vae_path = os.path.join(flux2_path, "vae")
    logger.info(f"Loading FLUX.2 VAE from {vae_path} ...")
    vae = AutoencoderKLFlux2.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
    vae = vae.to("cuda").eval()

    with torch.no_grad():
        image_bf16 = image.to(dtype=torch.bfloat16, device="cuda")
        latent = vae.encode(image_bf16).latent_dist.sample()
        # latent is already in BN-normalized, patchified space: [B, 128, H/16, W/16]

    logger.info(f"Encoded latent: {list(latent.shape)} "
                f"(mean={latent.mean():.3f}, std={latent.std():.3f})")

    # Also decode for baseline
    with torch.no_grad():
        vae_decoded = vae.decode(latent, return_dict=False)[0]

    # Free VAE
    del vae
    torch.cuda.empty_cache()

    return latent, vae_decoded


# ─── Option B: Use PiD's built-in VAE ───────────────────────────────────────

def encode_with_pid_vae(image: torch.Tensor, pid_model) -> torch.Tensor:
    """Encode image using the VAE that ships inside the PiD model."""
    if pid_model.vae_encoder is None:
        logger.error("PiD model has no VAE encoder loaded. Check tokenizer config.")
        sys.exit(1)

    image_bf16 = image.to(dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        latent = pid_model.encode_lq_latent(image_bf16)  # [B, C, zH, zW]

    logger.info(f"Encoded latent via PiD VAE: {list(latent.shape)} "
                f"(mean={latent.mean():.3f}, std={latent.std():.3f})")

    # VAE decode for baseline
    with torch.no_grad():
        z5 = latent.unsqueeze(2)  # [B, C, 1, zH, zW]
        recon5 = pid_model.vae_encoder.decode(z5)
        if recon5.ndim == 5:
            recon5 = recon5[:, :, 0]
        vae_decoded = recon5

    return latent, vae_decoded


# ─── Load PiD ────────────────────────────────────────────────────────────────

def load_pid_model(checkpoint_path: str, experiment: str, config_file: str):
    """Load PiD pixel decoder."""
    from pid._src.utils.model_loader import load_model_from_checkpoint

    logger.info(f"Loading PiD decoder from {checkpoint_path} ...")
    model, config = load_model_from_checkpoint(
        experiment_name=experiment,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        enable_fsdp=False,
        strict=False,
    )
    model.eval()
    logger.info("PiD decoder ready.")
    return model


# ─── PiD decode ──────────────────────────────────────────────────────────────

def pid_decode(model, latent: torch.Tensor, prompt: str,
               sigma: float, scale: int, pid_steps: int,
               pid_cfg_scale: float, seed: int) -> torch.Tensor:
    """Run PiD on a VAE latent. Returns [B, C, 1, H_out, W_out]."""
    B = latent.shape[0]
    # FLUX.2 VAE: 16× spatial compression
    vae_compression = 16
    lq_h = latent.shape[-2] * vae_compression
    lq_w = latent.shape[-1] * vae_compression
    target_h = lq_h * scale
    target_w = lq_w * scale

    # PiD condition: latent + caption + sigma
    lq_placeholder = torch.zeros(B, 3, lq_h, lq_w, dtype=torch.bfloat16, device="cuda")

    data_batch = {
        model.config.input_caption_key: [prompt] * B,
        "LQ_video_or_image": lq_placeholder,
        "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([sigma] * B, device="cuda", dtype=torch.float32),
    }

    logger.info(f"PiD: {list(latent.shape)} → ({target_h}×{target_w}), "
                f"sigma={sigma:.3f}, {pid_steps} steps")

    with torch.no_grad():
        samples = model.generate_samples_from_batch(
            data_batch,
            cfg_scale=pid_cfg_scale,
            num_steps=pid_steps,
            seed=seed,
            image_size=(target_h, target_w),
        )
    return samples


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.enable_grad(False)

    basename = os.path.splitext(os.path.basename(args.input_path))[0]

    # ---- Check PiD checkpoint exists ----
    if not os.path.exists(args.pid_checkpoint):
        logger.error(
            f"PiD checkpoint not found: {args.pid_checkpoint}\n"
            f"Download it:\n"
            f"  bash download_checkpoints.sh flux2"
        )
        sys.exit(1)

    # ---- Load input image ----
    image = load_image(args.input_path, args.resolution)
    logger.info(f"Input image: {args.input_path} → {list(image.shape)}")

    # Save input
    input_pil = tensor_to_pil(image[0])
    input_pil.save(os.path.join(args.output_dir, f"{basename}_input.jpg"), quality=95)

    if args.use_pid_vae:
        # ---- Path B: Load PiD first (it has its own VAE), encode with it ----
        pid_model = load_pid_model(args.pid_checkpoint, PID_EXPERIMENT, PID_CONFIG)
        latent, vae_decoded = encode_with_pid_vae(image, pid_model)
    else:
        # ---- Path A: Encode with diffusers VAE, then load PiD ----
        latent, vae_decoded = encode_with_diffusers_vae(image, args.flux2_path)
        pid_model = load_pid_model(args.pid_checkpoint, PID_EXPERIMENT, PID_CONFIG)

    # ---- Save VAE reconstruction (baseline) ----
    vae_pil = tensor_to_pil(vae_decoded[0].float().cpu().clamp(-1, 1))
    vae_path = os.path.join(args.output_dir, f"{basename}_vae_recon.jpg")
    vae_pil.save(vae_path, quality=95)
    logger.info(f"VAE reconstruction: {vae_path} ({vae_pil.size})")

    # ---- Add noise if requested ----
    if args.sigma > 0:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        noise = torch.randn(latent.shape, generator=gen, device=latent.device, dtype=latent.dtype)
        latent = (1.0 - args.sigma) * latent + args.sigma * noise
        logger.info(f"Added noise: sigma={args.sigma:.3f}")

    # ---- PiD decode ----
    samples = pid_decode(
        pid_model, latent, args.prompt,
        sigma=args.sigma, scale=args.scale,
        pid_steps=args.pid_steps,
        pid_cfg_scale=args.pid_cfg_scale,
        seed=args.seed,
    )

    # ---- Save PiD output ----
    pid_img = samples[0].float().cpu().clamp(-1, 1)
    if pid_img.dim() == 4:
        pid_img = pid_img.squeeze(1)  # remove T dim [C, 1, H, W] → [C, H, W]
    pid_pil = tensor_to_pil(pid_img)
    pid_path = os.path.join(args.output_dir, f"{basename}_pid_sr.jpg")
    pid_pil.save(pid_path, quality=95)
    logger.info(f"PiD SR output: {pid_path} ({pid_pil.size})")

    # ---- Side-by-side: input | VAE recon | PiD SR ----
    target_h = pid_pil.height
    input_resized = input_pil.resize((target_h, target_h), Image.BICUBIC)
    vae_resized = vae_pil.resize((target_h, target_h), Image.BICUBIC)

    comparison = Image.new("RGB", (target_h * 3 + 20, target_h))
    comparison.paste(input_resized, (0, 0))
    comparison.paste(vae_resized, (target_h + 10, 0))
    comparison.paste(pid_pil, (target_h * 2 + 20, 0))
    cmp_path = os.path.join(args.output_dir, f"{basename}_comparison.jpg")
    comparison.save(cmp_path, quality=95)
    logger.info(f"Comparison (input | VAE | PiD): {cmp_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
