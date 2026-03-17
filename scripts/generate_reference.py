#!/usr/bin/env python3
"""Generate reference tensors for validating diffusers-burn against HuggingFace diffusers.

Runs the Stable Diffusion pipeline step-by-step and saves intermediate tensors
as safetensors files. These can be compared against diffusers-burn output to
verify numerical correctness.

Requires: torch, diffusers, transformers, safetensors, Pillow

Usage:
    python scripts/generate_reference.py --version v1.5 --output-dir output/ref_sd15
    python scripts/generate_reference.py --version v2.1 --output-dir output/ref_sd21
    python scripts/generate_reference.py --version xl --output-dir output/ref_sdxl
    python scripts/generate_reference.py --version xl --load-latents output/latents.safetensors
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

PROMPT = "a green apple on a wooden table"
NEGATIVE_PROMPT = ""
GUIDANCE_SCALE = 7.5

CONFIGS = {
    "v1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "height": 512,
        "width": 512,
        "prediction_type": "epsilon",
        "xl": False,
        "vae_scale": 0.18215,
    },
    "v2.1": {
        "model_id": "sd2-community/stable-diffusion-2-1",
        "height": 768,
        "width": 768,
        "prediction_type": "v_prediction",
        "xl": False,
        "vae_scale": 0.18215,
    },
    "xl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "height": 1024,
        "width": 1024,
        "prediction_type": "epsilon",
        "xl": True,
        "vae_scale": 0.13025,
    },
}


def main(output_dir: str, version: str, load_latents: str = None,
         seed: int = 42, n_steps: int = 20, device: str = "cpu"):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    cfg = CONFIGS[version]
    model_id = cfg["model_id"]
    height, width = cfg["height"], cfg["width"]
    is_xl = cfg["xl"]
    vae_scale = cfg["vae_scale"]

    os.makedirs(output_dir, exist_ok=True)
    dtype = torch.float32

    # -- Load models --
    print(f"Loading {version} models from {model_id} (device={device})...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)

    if is_xl:
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(device)

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        steps_offset=1,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=cfg["prediction_type"],
    )
    scheduler.set_timesteps(n_steps, device=device)

    # -- Tokenize --
    print("Tokenizing...")
    text_input = tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    )
    uncond_input = tokenizer(
        NEGATIVE_PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    )

    save_file(
        {"prompt_tokens": text_input.input_ids.int(), "uncond_tokens": uncond_input.input_ids.int()},
        os.path.join(output_dir, "tokens.safetensors"),
    )

    # -- CLIP embeddings --
    print("Encoding text...")
    with torch.no_grad():
        if is_xl:
            text_input_2 = tokenizer_2(
                PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
            )

            enc1_out = text_encoder(text_input.input_ids.to(device), output_hidden_states=True)
            text_emb_1 = enc1_out.hidden_states[-2]

            enc2_out = text_encoder_2(text_input_2.input_ids.to(device), output_hidden_states=True)
            text_emb_2 = enc2_out.hidden_states[-2]
            text_pooled = enc2_out[0]

            text_embeddings = torch.cat([text_emb_1, text_emb_2], dim=-1)

            # SDXL: force zeros for empty/negative prompt
            uncond_embeddings = torch.zeros_like(text_embeddings)
            uncond_pooled = torch.zeros_like(text_pooled)

            save_file(
                {"text_embeddings": text_embeddings, "uncond_embeddings": uncond_embeddings,
                 "text_pooled": text_pooled, "uncond_pooled": uncond_pooled},
                os.path.join(output_dir, "clip_embeddings.safetensors"),
            )

            combined = torch.cat([uncond_embeddings, text_embeddings])
            pooled = torch.cat([uncond_pooled, text_pooled])

            time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype, device=device)
            time_ids = torch.cat([time_ids, time_ids])

            added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}
        else:
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

            save_file(
                {"text_embeddings": text_embeddings, "uncond_embeddings": uncond_embeddings},
                os.path.join(output_dir, "clip_embeddings.safetensors"),
            )

            combined = torch.cat([uncond_embeddings, text_embeddings])
            added_cond_kwargs = None

    save_file(
        {"combined_embeddings": combined},
        os.path.join(output_dir, "combined_embeddings.safetensors"),
    )

    # -- Initial latents --
    if load_latents:
        print(f"Loading initial latents from {load_latents}")
        from safetensors.torch import load_file as load_st
        latents = load_st(load_latents)["initial_latents"].to(device=device, dtype=dtype)
    else:
        print(f"Generating initial latents (seed={seed})...")
        torch.manual_seed(seed)
        latents = torch.randn((1, 4, height // 8, width // 8), device=device, dtype=dtype)
    save_file({"initial_latents": latents}, os.path.join(output_dir, "initial_latents.safetensors"))

    latents = latents * scheduler.init_noise_sigma
    save_file({"scaled_latents": latents}, os.path.join(output_dir, "scaled_latents.safetensors"))
    save_file({"timesteps": scheduler.timesteps.float()}, os.path.join(output_dir, "timesteps.safetensors"))

    # -- Denoising loop --
    print(f"Denoising ({n_steps} steps)...")
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            unet_kwargs = {"encoder_hidden_states": combined}
            if added_cond_kwargs is not None:
                unet_kwargs["added_cond_kwargs"] = added_cond_kwargs

            noise_pred = unet(latent_model_input, t, **unet_kwargs).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(guided, t, latents).prev_sample

            if i < 3 or i >= n_steps - 2 or i % 5 == 0:
                save_file(
                    {"noise_pred": guided, "latents": latents},
                    os.path.join(output_dir, f"step_{i:03d}.safetensors"),
                )
            print(f"  Step {i:2d}/{n_steps}: t={t.item():4.0f}  "
                  f"mean={latents.mean().item():+.6f} std={latents.std().item():.6f}")

    save_file({"final_latents": latents}, os.path.join(output_dir, "final_latents.safetensors"))

    # -- Decode --
    print("Decoding...")
    with torch.no_grad():
        decoded = vae.decode(latents / vae_scale).sample
        image = (decoded / 2 + 0.5).clamp(0, 1)

    save_file({"decoded_image": image}, os.path.join(output_dir, "decoded_image.safetensors"))

    img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(img_np).save(os.path.join(output_dir, "reference.png"))

    print(f"\nSaved to {output_dir}/")
    print(f"  reference.png + {len(os.listdir(output_dir))-1} safetensors files")
    print(f"\nConfig: prompt='{PROMPT}', seed={seed}, steps={n_steps}, "
          f"guidance={GUIDANCE_SCALE}, size={width}x{height}, device={device}")
    print(f"Final latents: mean={latents.mean().item():+.6f}, std={latents.std().item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SD reference tensors for diffusers-burn validation")
    parser.add_argument("--output-dir", required=True, help="Directory to save reference tensors")
    parser.add_argument("--version", required=True, choices=list(CONFIGS.keys()), help="SD version")
    parser.add_argument("--load-latents", help="Load initial latents from safetensors file instead of generating")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps (default: 20)")
    parser.add_argument("--device", default="cpu", help="Device to run on (default: cpu)")
    args = parser.parse_args()
    main(args.output_dir, args.version, args.load_latents, args.seed, args.steps, args.device)
