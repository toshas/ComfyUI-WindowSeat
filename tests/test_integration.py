"""Integration tests for WindowSeat.

These tests require a GPU and will load the actual model.
Run with: pytest tests/test_integration.py -v --gpu
"""

import gc
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def loaded_model():
    """
    Module-scoped fixture to load the model once for all tests.
    This avoids loading the model multiple times which wastes memory.
    """
    from windowseat_core import load_network

    clear_cuda_cache()
    device = torch.device("cuda")
    model = load_network(device=device)

    yield model

    # Cleanup after all tests
    del model
    clear_cuda_cache()


@pytest.mark.gpu
class TestModelLoading:
    """Integration tests for model loading."""

    def test_load_network(self, loaded_model):
        """Test that the network loads successfully."""
        model = loaded_model

        assert model is not None
        assert len(model) == 4  # (vae, transformer, embeds_dict, processing_resolution)

        vae, transformer, embeds_dict, processing_resolution = model
        assert vae is not None
        assert transformer is not None
        assert embeds_dict is not None
        assert processing_resolution == 768

    def test_node_loader_uses_cache(self, loaded_model):
        """Test that the ComfyUI node loader uses cached model."""
        from nodes import WindowSeatModelLoader

        # Pre-populate cache
        WindowSeatModelLoader._cached_model = loaded_model
        WindowSeatModelLoader._cached_device = torch.device("cuda")

        loader = WindowSeatModelLoader()
        result = loader.load_model()

        assert len(result) == 1
        model = result[0]
        assert len(model) == 4
        assert model is loaded_model  # Should be the same cached model


@pytest.mark.gpu
class TestImageProcessing:
    """Integration tests for image processing."""

    def test_process_small_image(self, loaded_model):
        """Test processing a small image."""
        from windowseat_core import process_image

        clear_cuda_cache()

        # Create small test image (256x256 to minimize memory)
        image = torch.randn(3, 256, 256)
        result = process_image(loaded_model, image, tile_batch_size=1)

        assert result.shape == image.shape
        assert result.dtype == torch.float32
        assert result.min() >= -1.0
        assert result.max() <= 1.0

        clear_cuda_cache()

    def test_process_medium_image(self, loaded_model):
        """Test processing a medium image."""
        from windowseat_core import process_image

        clear_cuda_cache()

        # Create medium test image (512x512)
        image = torch.randn(3, 512, 512)
        result = process_image(loaded_model, image, tile_batch_size=1)

        assert result.shape == image.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        clear_cuda_cache()


@pytest.mark.gpu
class TestComfyUINode:
    """Integration tests for the ComfyUI node."""

    def test_reflection_removal_single_image(self, loaded_model):
        """Test reflection removal on single image."""
        from nodes import WindowSeatModelLoader, WindowSeatReflectionRemoval

        clear_cuda_cache()

        # Set up model cache
        WindowSeatModelLoader._cached_model = loaded_model
        WindowSeatModelLoader._cached_device = torch.device("cuda")

        node = WindowSeatReflectionRemoval()

        # ComfyUI format: [B, H, W, C] in [0, 1] - small image
        image = torch.rand(1, 256, 256, 3)

        result = node.remove_reflections(loaded_model, image, tile_batch_size=1)

        assert len(result) == 1
        output = result[0]
        assert output.shape == image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0

        clear_cuda_cache()


@pytest.mark.gpu
class TestWithExampleImages:
    """Tests using example images if available."""

    def test_example_image(self, loaded_model):
        """Test with an example image if available."""
        from windowseat_core import process_pil_image

        clear_cuda_cache()

        # Look for example images
        example_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_images"),
        ]

        example_image = None
        for example_dir in example_dirs:
            if os.path.isdir(example_dir):
                for f in os.listdir(example_dir):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        example_image = os.path.join(example_dir, f)
                        break
            if example_image:
                break

        if example_image is None:
            pytest.skip("No example images found")

        # Load and resize to manageable size
        pil_image = Image.open(example_image).convert("RGB")
        # Resize to max 512 on long edge to avoid OOM
        max_size = 512
        w, h = pil_image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        result = process_pil_image(loaded_model, pil_image, tile_batch_size=1)

        assert isinstance(result, Image.Image)
        assert result.size == pil_image.size

        clear_cuda_cache()
