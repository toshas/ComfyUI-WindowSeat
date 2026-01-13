"""
WindowSeat Reflection Removal - Core Inference Logic

Adapted from https://github.com/huawei-bayerlab/windowseat-reflection-removal
"""

import json
import math

import numpy as np
import safetensors
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from PIL import Image

# Model URIs
SUPPORTED_MODEL_URIS = ["Qwen/Qwen-Image-Edit-2509"]
DEFAULT_BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_LORA_MODEL_URI = "huawei-bayerlab/windowseat-reflection-removal-v1-0"


def fetch_state_dict(
    pretrained_model_name_or_path_or_dict: str,
    weight_name: str,
    use_safetensors: bool = True,
    subfolder: str | None = None,
) -> dict:
    """Fetch state dict from HuggingFace Hub."""
    file_path = hf_hub_download(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder=subfolder
    )
    if use_safetensors:
        state_dict = safetensors.torch.load_file(file_path)
    else:
        state_dict = torch.load(file_path, weights_only=True)
    return state_dict


def load_qwen_vae(uri: str, device: torch.device) -> AutoencoderKLQwenImage:
    """Load Qwen VAE model."""
    vae = AutoencoderKLQwenImage.from_pretrained(
        uri,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    return vae


def load_qwen_transformer(uri: str, device: torch.device) -> QwenImageTransformer2DModel:
    """Load Qwen transformer with NF4 quantization."""
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )

    transformer = QwenImageTransformer2DModel.from_pretrained(
        uri,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4,
        device_map=device,
    )

    return transformer


def load_lora_into_transformer(
    uri: str, transformer: QwenImageTransformer2DModel
) -> QwenImageTransformer2DModel:
    """Load LoRA adapter into transformer."""
    lora_config = LoraConfig.from_pretrained(uri, subfolder="transformer_lora")
    transformer.add_adapter(lora_config)
    state_dict = fetch_state_dict(
        uri, "pytorch_lora_weights.safetensors", subfolder="transformer_lora"
    )
    _missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected keys in transformer state dict: {unexpected}")
    return transformer


def load_embeds_dict(uri: str) -> dict:
    """Load precomputed text embeddings."""
    embeds_dict = fetch_state_dict(uri, "state_dict.safetensors", subfolder="text_embeddings")
    return embeds_dict


def load_network(
    uri_base: str = DEFAULT_BASE_MODEL_URI,
    uri_lora: str = DEFAULT_LORA_MODEL_URI,
    device: torch.device | None = None,
) -> tuple:
    """
    Load the complete WindowSeat network.

    Returns:
        tuple: (vae, transformer, embeds_dict, processing_resolution)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_file = hf_hub_download(uri_lora, "model_index.json")
    with open(config_file) as f:
        config_dict = json.load(f)

    base_model_uri = config_dict["base_model"]
    processing_resolution = config_dict["processing_resolution"]

    if base_model_uri not in SUPPORTED_MODEL_URIS:
        raise ValueError(f"Unsupported base model URI: {base_model_uri}")

    vae = load_qwen_vae(uri_base, device)
    transformer = load_qwen_transformer(uri_base, device)
    load_lora_into_transformer(uri_lora, transformer)
    embeds_dict = load_embeds_dict(uri_lora)

    return vae, transformer, embeds_dict, processing_resolution


def encode(image: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    """Encode image to latent space."""
    image = image.to(device=vae.device, dtype=vae.dtype)
    out = vae.encode(image.unsqueeze(2)).latent_dist.sample()
    latents_mean = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=out.device, dtype=out.dtype)
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    out = (out - latents_mean) * latents_std_inv
    return out


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    """Decode latents to image."""
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(
        vae.config.latents_std, device=latents.device, dtype=latents.dtype
    )
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean
    out = vae.decode(latents)
    out = out.sample[:, :, 0]
    return out


def _match_batch(t: torch.Tensor, B: int) -> torch.Tensor:
    """Match tensor batch dimension."""
    if t.size(0) == B:
        return t
    if t.size(0) == 1 and B > 1:
        return t.expand(B, *t.shape[1:])
    if t.size(0) > B:
        return t[:B]
    reps = (B + t.size(0) - 1) // t.size(0)
    return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]


def flow_step(
    model_input: torch.Tensor,
    transformer: QwenImageTransformer2DModel,
    vae: AutoencoderKLQwenImage,
    embeds_dict: dict,
) -> torch.Tensor:
    """Run single flow step through transformer."""
    prompt_embeds = embeds_dict["prompt_embeds"]
    prompt_mask = embeds_dict["prompt_mask"]

    if prompt_mask.dtype != torch.bool:
        prompt_mask = prompt_mask > 0

    # Handle input shape
    if model_input.ndim == 5 and model_input.shape[2] == 1:
        model_input_4d = model_input[:, :, 0]
    elif model_input.ndim == 4:
        model_input_4d = model_input
    else:
        raise ValueError(f"Unexpected lat_encoding shape: {model_input.shape}")

    B, C, H, W = model_input_4d.shape
    device = next(transformer.parameters()).device

    prompt_embeds = _match_batch(prompt_embeds, B).to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )
    prompt_mask = _match_batch(prompt_mask, B).to(
        device=device, dtype=torch.bool, non_blocking=True
    )

    num_channels_latents = C
    packed_model_input = QwenImageEditPipeline._pack_latents(
        model_input_4d,
        batch_size=B,
        num_channels_latents=num_channels_latents,
        height=H,
        width=W,
    )
    packed_model_input = packed_model_input.to(torch.bfloat16)

    t_const = 499
    timestep = torch.full((B,), float(t_const), device=device, dtype=torch.bfloat16)
    timestep = timestep / 1000.0

    h_img = H // 2
    w_img = W // 2

    img_shapes = [[(1, h_img, w_img)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist() if prompt_mask is not None else None

    attention_kwargs = getattr(transformer, "attention_kwargs", {}) or {}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    temperal_downsample = vae.config.get("temperal_downsample", None)
    if temperal_downsample is not None:
        vae_scale_factor = 2 ** len(temperal_downsample)
    else:
        vae_scale_factor = 8

    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred,
        height=H * vae_scale_factor,
        width=W * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    latent_output = model_input.to(vae.dtype) - model_pred.to(vae.dtype)
    return latent_output


def _required_side_for_axis(size: int, nmax: int, min_overlap: int) -> int:
    """Smallest tile side T (1D) so that #tiles <= nmax with overlap >= min_overlap."""
    nmax = max(1, int(nmax))
    if nmax == 1:
        return size
    return math.ceil((size + (nmax - 1) * min_overlap) / nmax)


def _compute_starts(size: int, T: int, min_overlap: int) -> list:
    """Uniform stepping with stride = T - min_overlap; last tile flush with edge."""
    if size <= T:
        return [0]
    stride = max(1, T - min_overlap)
    xs = list(range(0, size - T + 1, stride))
    last = size - T
    if xs[-1] != last:
        xs.append(last)
    # monotonic dedupe
    out = []
    for v in xs:
        if not out or v > out[-1]:
            out.append(v)
    return out


def compute_tiles(
    W: int,
    H: int,
    tiling_size: int = 768,
    max_tiles_w: int = 4,
    max_tiles_h: int = 4,
    min_overlap: int = 64,
    use_short_edge_tile: bool = True,
    processing_resolution: int = 768,
) -> tuple:
    """
    Compute tile grid for an image.

    Args:
        W: Image width
        H: Image height
        tiling_size: Base tile size
        max_tiles_w: Max tiles horizontally
        max_tiles_h: Max tiles vertically
        min_overlap: Minimum overlap between tiles
        use_short_edge_tile: If True, use short edge for tile size
        processing_resolution: Minimum processing resolution

    Returns:
        tuple: (tiles_list, scale_ratio) where tiles_list is [(x0, y0, x1, y1), ...]
               and scale_ratio is the factor to scale the image before tiling
    """
    # Determine tile size
    if use_short_edge_tile:
        short_edge = min(W, H)
        short_edge = max(short_edge, processing_resolution)
        pref_tile = short_edge
    else:
        pref_tile = tiling_size

    # Compute scale ratio if image is smaller than tile
    scale_ratio = 1.0
    scaled_W, scaled_H = W, H
    if pref_tile > W or pref_tile > H:
        min_side = min(W, H)
        scale_ratio = pref_tile / min_side
        scaled_W = round(scale_ratio * W)
        scaled_H = round(scale_ratio * H)

    Nw, Nh = int(max_tiles_w), int(max_tiles_h)
    ow, oh = int(min_overlap), int(min_overlap)

    # Feasible square-side interval
    T_low = max(
        _required_side_for_axis(scaled_W, Nw, ow),
        _required_side_for_axis(scaled_H, Nh, oh),
        ow + 1,
        oh + 1,
    )
    T_high = min(scaled_W, scaled_H)

    if T_low > T_high:
        raise ValueError(
            f"Infeasible tile constraints: need T >= {T_low}, but max is {T_high}. "
            f"Try increasing max_tiles_w/h or decreasing min_overlap."
        )

    T = max(T_low, min(pref_tile, T_high))

    # Compute tile positions
    xs = _compute_starts(scaled_W, T, ow)
    ys = _compute_starts(scaled_H, T, oh)

    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((x0, y0, x0 + T, y0 + T))

    return tiles, scale_ratio, (scaled_W, scaled_H)


def _lanczos_resize_chw(x: torch.Tensor, out_hw: tuple) -> torch.Tensor:
    """Resize CHW tensor using Lanczos interpolation."""
    H_out, W_out = map(int, out_hw)

    dev = x.device
    arr = x.detach().cpu().numpy()

    assert arr.ndim == 3, "expect CHW"
    chw = arr.astype(np.float32, copy=False)
    C, _, _ = chw.shape

    out_chw = np.empty((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        ch = chw[c]
        img = Image.fromarray(ch).convert("F")
        img = img.resize((W_out, H_out), resample=Image.LANCZOS)
        out_chw[c] = np.asarray(img, dtype=np.float32)

    return torch.from_numpy(out_chw).to(dev)


def stitch_tiles(
    tiles_data: list,
    output_W: int,
    output_H: int,
) -> torch.Tensor:
    """
    Stitch tiles together using triangular window blending.

    Args:
        tiles_data: List of (tile_tensor, (x0, y0, x1, y1)) tuples
        output_W: Output width
        output_H: Output height

    Returns:
        torch.Tensor: Stitched image [3, H, W] in [-1, 1] range
    """
    acc = torch.zeros(3, output_H, output_W, dtype=torch.float32)
    wsum = torch.zeros(output_H, output_W, dtype=torch.float32)

    for tile, (x0, y0, x1, y1) in tiles_data:
        tile = tile.squeeze(0).float().cpu()  # [3, h, w]

        h, w = tile.shape[-2:]
        tH, tW = (y1 - y0), (x1 - x0)

        if (h != tH) or (w != tW):
            tile = _lanczos_resize_chw(tile, (tH, tW))
            h, w = tH, tW

        # Triangular window
        wx = 1 - (2 * torch.arange(w, dtype=torch.float32) / (max(w - 1, 1)) - 1).abs()
        wy = 1 - (2 * torch.arange(h, dtype=torch.float32) / (max(h - 1, 1)) - 1).abs()
        w2 = (wy[:, None] * wx[None, :]).clamp_min(1e-3)

        acc[:, y0:y1, x0:x1] += tile * w2
        wsum[y0:y1, x0:x1] += w2

    stitched = acc / wsum.clamp_min(1e-6)
    return stitched


@torch.no_grad()
def process_image(
    model: tuple,
    image: torch.Tensor,
    use_short_edge_tile: bool = True,
    tiling_size: int = 768,
    max_tiles_w: int = 4,
    max_tiles_h: int = 4,
    min_overlap: int = 64,
    tile_batch_size: int = 2,
) -> torch.Tensor:
    """
    Process a single image through WindowSeat reflection removal.

    Args:
        model: Tuple of (vae, transformer, embeds_dict, processing_resolution)
        image: Input image tensor [C, H, W] in [-1, 1] range
        use_short_edge_tile: Use short edge for tile size
        tiling_size: Base tile size
        max_tiles_w: Max tiles horizontally
        max_tiles_h: Max tiles vertically
        min_overlap: Minimum overlap between tiles
        tile_batch_size: Number of tiles to process at once

    Returns:
        torch.Tensor: Processed image [C, H, W] in [-1, 1] range
    """
    vae, transformer, embeds_dict, processing_resolution = model

    # Get original dimensions
    _C, orig_H, orig_W = image.shape

    # Compute tiles
    tiles, scale_ratio, (scaled_W, scaled_H) = compute_tiles(
        orig_W,
        orig_H,
        tiling_size=tiling_size,
        max_tiles_w=max_tiles_w,
        max_tiles_h=max_tiles_h,
        min_overlap=min_overlap,
        use_short_edge_tile=use_short_edge_tile,
        processing_resolution=processing_resolution,
    )

    # Scale image if needed
    if scale_ratio != 1.0:
        scaled_image = _lanczos_resize_chw(image, (scaled_H, scaled_W))
    else:
        scaled_image = image

    # Process tiles
    tiles_data = []

    for i in range(0, len(tiles), tile_batch_size):
        batch_tiles = tiles[i : i + tile_batch_size]
        batch_inputs = []

        for x0, y0, x1, y1 in batch_tiles:
            tile_crop = scaled_image[:, y0:y1, x0:x1]
            # Resize to processing resolution
            tile_resized = _lanczos_resize_chw(
                tile_crop, (processing_resolution, processing_resolution)
            )
            batch_inputs.append(tile_resized)

        # Stack batch
        batch_tensor = torch.stack(batch_inputs, dim=0)  # [B, C, H, W]

        # Encode
        latents = encode(batch_tensor, vae)

        # Flow step
        latents = flow_step(latents, transformer, vae, embeds_dict)

        # Decode
        decoded = decode(latents, vae)  # [B, C, H, W]

        # Store results
        for j, (x0, y0, x1, y1) in enumerate(batch_tiles):
            tiles_data.append((decoded[j : j + 1], (x0, y0, x1, y1)))

    # Stitch tiles
    stitched = stitch_tiles(tiles_data, scaled_W, scaled_H)

    # Resize back to original
    if scale_ratio != 1.0:
        result = _lanczos_resize_chw(stitched, (orig_H, orig_W))
    else:
        result = stitched

    return result.clamp(-1, 1)


def process_pil_image(
    model: tuple,
    pil_image: Image.Image,
    **kwargs,
) -> Image.Image:
    """
    Process a PIL image through WindowSeat.

    Args:
        model: Tuple of (vae, transformer, embeds_dict, processing_resolution)
        pil_image: Input PIL image
        **kwargs: Additional arguments for process_image

    Returns:
        PIL.Image: Processed image
    """
    # Convert PIL to tensor
    arr = np.array(pil_image.convert("RGB"), dtype=np.float32)  # [H, W, 3]
    arr = arr / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    # Process
    result = process_image(model, tensor, **kwargs)

    # Convert back to PIL
    result_np = result.permute(1, 2, 0).numpy()  # [H, W, 3]
    result_np = ((result_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(result_np)
