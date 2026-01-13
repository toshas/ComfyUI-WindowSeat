"""
ComfyUI nodes for WindowSeat Reflection Removal.

Two nodes are provided:
1. WindowSeatModelLoader - Loads and caches the model
2. WindowSeatReflectionRemoval - Processes images to remove reflections
"""

import torch

try:
    from .windowseat_core import load_network, process_image
except ImportError:
    from windowseat_core import load_network, process_image


class WindowSeatModelLoader:
    """
    Load the WindowSeat reflection removal model.

    Downloads models from HuggingFace Hub on first use and caches them.
    """

    # Class-level cache for model
    _cached_model = None
    _cached_device = None

    CATEGORY = "WindowSeat"
    FUNCTION = "load_model"
    RETURN_TYPES = ("WINDOWSEAT_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    def load_model(self):
        """Load or return cached model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check cache
        if (
            WindowSeatModelLoader._cached_model is not None
            and WindowSeatModelLoader._cached_device == device
        ):
            return (WindowSeatModelLoader._cached_model,)

        # Load model
        model = load_network(device=device)

        # Cache
        WindowSeatModelLoader._cached_model = model
        WindowSeatModelLoader._cached_device = device

        return (model,)


class WindowSeatReflectionRemoval:
    """
    Remove reflections from images using WindowSeat.

    Supports batched inputs and tiled processing for high-resolution images.
    """

    CATEGORY = "WindowSeat"
    FUNCTION = "remove_reflections"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WINDOWSEAT_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "use_short_edge_tile": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "tiling_size": (
                    "INT",
                    {"default": 768, "min": 512, "max": 1536, "step": 64},
                ),
                "max_tiles_w": (
                    "INT",
                    {"default": 4, "min": 1, "max": 8, "step": 1},
                ),
                "max_tiles_h": (
                    "INT",
                    {"default": 4, "min": 1, "max": 8, "step": 1},
                ),
                "min_overlap": (
                    "INT",
                    {"default": 64, "min": 16, "max": 256, "step": 16},
                ),
                "tile_batch_size": (
                    "INT",
                    {"default": 2, "min": 1, "max": 4, "step": 1},
                ),
            },
        }

    def remove_reflections(
        self,
        model,
        image,
        use_short_edge_tile=True,
        tiling_size=768,
        max_tiles_w=4,
        max_tiles_h=4,
        min_overlap=64,
        tile_batch_size=2,
    ):
        """
        Remove reflections from input images.

        Args:
            model: WindowSeat model tuple from loader
            image: Input tensor [B, H, W, C] in [0, 1] range (ComfyUI format)
            use_short_edge_tile: Use short edge for tile size
            tiling_size: Base tile size
            max_tiles_w: Max tiles horizontally
            max_tiles_h: Max tiles vertically
            min_overlap: Minimum tile overlap
            tile_batch_size: Tiles to process per batch

        Returns:
            tuple: (output_tensor,) where output is [B, H, W, C] in [0, 1]
        """
        # Get batch size
        batch_size = image.shape[0]
        results = []

        # Process each image in the batch
        for i in range(batch_size):
            # Convert ComfyUI format [H, W, C] [0, 1] to WindowSeat [C, H, W] [-1, 1]
            single_image = image[i]  # [H, W, C]
            img_tensor = single_image.permute(2, 0, 1)  # [C, H, W]
            img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

            # Process
            result = process_image(
                model,
                img_tensor,
                use_short_edge_tile=use_short_edge_tile,
                tiling_size=tiling_size,
                max_tiles_w=max_tiles_w,
                max_tiles_h=max_tiles_h,
                min_overlap=min_overlap,
                tile_batch_size=tile_batch_size,
            )

            # Convert back to ComfyUI format
            result = (result + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            result = result.clamp(0, 1)
            result = result.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

            results.append(result)

        # Stack results
        output = torch.stack(results, dim=0)  # [B, H, W, C]

        return (output,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WindowSeatModelLoader": WindowSeatModelLoader,
    "WindowSeatReflectionRemoval": WindowSeatReflectionRemoval,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WindowSeatModelLoader": "WindowSeat Model Loader",
    "WindowSeatReflectionRemoval": "WindowSeat Reflection Removal",
}
