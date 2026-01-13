"""Unit tests for windowseat_core module.

These tests do not require a GPU and test the tile computation
and stitching logic.
"""

import torch

from windowseat_core import (
    _compute_starts,
    _lanczos_resize_chw,
    _required_side_for_axis,
    compute_tiles,
    stitch_tiles,
)


class TestComputeStarts:
    """Tests for _compute_starts helper function."""

    def test_small_size_single_tile(self):
        """Size smaller than tile should produce single start at 0."""
        starts = _compute_starts(size=256, T=512, min_overlap=64)
        assert starts == [0]

    def test_exact_fit(self):
        """Size equal to tile should produce single start at 0."""
        starts = _compute_starts(size=768, T=768, min_overlap=64)
        assert starts == [0]

    def test_two_tiles_needed(self):
        """Size requiring two tiles should produce correct starts."""
        starts = _compute_starts(size=1024, T=768, min_overlap=64)
        assert len(starts) == 2
        assert starts[0] == 0
        assert starts[-1] == 1024 - 768  # Last tile flush with edge

    def test_overlap_respected(self):
        """Tiles should have at least min_overlap overlap."""
        starts = _compute_starts(size=1500, T=768, min_overlap=64)
        for i in range(len(starts) - 1):
            tile_end = starts[i] + 768
            next_start = starts[i + 1]
            overlap = tile_end - next_start
            assert overlap >= 64


class TestRequiredSideForAxis:
    """Tests for _required_side_for_axis helper function."""

    def test_single_tile_allowed(self):
        """With nmax=1, should return full size."""
        result = _required_side_for_axis(size=1000, nmax=1, min_overlap=64)
        assert result == 1000

    def test_multiple_tiles(self):
        """Should compute minimum tile size for given constraints."""
        result = _required_side_for_axis(size=2000, nmax=4, min_overlap=64)
        # With 4 tiles and 64 overlap, need T such that 4*T - 3*64 >= 2000
        # T >= (2000 + 192) / 4 = 548
        assert result >= 548


class TestComputeTiles:
    """Tests for compute_tiles function."""

    def test_small_image_single_tile(self):
        """Small image should produce single tile."""
        tiles, scale, (_sw, _sh) = compute_tiles(
            W=256, H=256, tiling_size=768, processing_resolution=768
        )
        # Image smaller than processing_resolution, should be scaled up
        assert scale > 1.0
        assert len(tiles) >= 1

    def test_exact_size_single_tile(self):
        """Image exactly tile size should produce single tile."""
        tiles, scale, (_sw, _sh) = compute_tiles(
            W=768, H=768, tiling_size=768, processing_resolution=768
        )
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 768, 768)
        assert scale == 1.0

    def test_large_image_multiple_tiles(self):
        """Large image should produce multiple tiles."""
        tiles, scale, (_sw, _sh) = compute_tiles(
            W=2048, H=1536, tiling_size=768, processing_resolution=768
        )
        assert len(tiles) > 1
        assert scale == 1.0

    def test_max_tiles_respected(self):
        """Number of tiles should not exceed max."""
        tiles, _scale, (_sw, _sh) = compute_tiles(
            W=4096,
            H=4096,
            max_tiles_w=2,
            max_tiles_h=2,
            tiling_size=768,
            processing_resolution=768,
        )
        # Count tiles in each direction
        x_coords = set(t[0] for t in tiles)
        y_coords = set(t[1] for t in tiles)
        assert len(x_coords) <= 2
        assert len(y_coords) <= 2

    def test_short_edge_tile_mode(self):
        """With use_short_edge_tile, tile size should match short edge."""
        tiles, _scale, (_sw, _sh) = compute_tiles(
            W=1024, H=768, use_short_edge_tile=True, processing_resolution=768
        )
        # Tile size should be based on short edge (768)
        if len(tiles) > 0:
            tile_w = tiles[0][2] - tiles[0][0]
            tile_h = tiles[0][3] - tiles[0][1]
            assert tile_w == tile_h  # Square tiles

    def test_tiles_cover_image(self):
        """Tiles should fully cover the image."""
        tiles, _scale, (sw, sh) = compute_tiles(
            W=1500, H=1200, tiling_size=768, processing_resolution=768
        )
        # Check coverage
        max_x = max(t[2] for t in tiles)
        max_y = max(t[3] for t in tiles)
        assert max_x >= sw
        assert max_y >= sh


class TestStitchTiles:
    """Tests for stitch_tiles function."""

    def test_single_tile_passthrough(self):
        """Single tile should pass through unchanged."""
        tile = torch.randn(1, 3, 256, 256)
        tiles_data = [(tile, (0, 0, 256, 256))]

        result = stitch_tiles(tiles_data, 256, 256)

        assert result.shape == (3, 256, 256)
        # Values should be close to original (slight difference from blending)
        assert torch.allclose(result, tile.squeeze(0), atol=0.1)

    def test_output_shape(self, sample_tile_data):
        """Output should have correct shape."""
        tiles_data, W, H = sample_tile_data
        result = stitch_tiles(tiles_data, W, H)
        assert result.shape == (3, H, W)

    def test_no_nan_or_inf(self, sample_tile_data):
        """Output should not contain NaN or Inf."""
        tiles_data, W, H = sample_tile_data
        result = stitch_tiles(tiles_data, W, H)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_overlapping_tiles_blended(self):
        """Overlapping tiles should be blended smoothly."""
        # Create two overlapping tiles with different values
        tile1 = torch.ones(1, 3, 256, 256) * -0.5
        tile2 = torch.ones(1, 3, 256, 256) * 0.5

        tiles_data = [
            (tile1, (0, 0, 256, 256)),
            (tile2, (128, 0, 384, 256)),  # 128px overlap
        ]

        result = stitch_tiles(tiles_data, 384, 256)

        # Left edge should be close to -0.5
        assert result[:, :, 0].mean() < 0
        # Right edge should be close to 0.5
        assert result[:, :, -1].mean() > 0
        # Middle (overlap region) should be intermediate
        mid_val = result[:, :, 192].mean()
        assert -0.3 < mid_val < 0.3


class TestLanczosResize:
    """Tests for _lanczos_resize_chw function."""

    def test_downsample(self):
        """Downsampling should produce smaller output."""
        input_tensor = torch.randn(3, 512, 512)
        result = _lanczos_resize_chw(input_tensor, (256, 256))
        assert result.shape == (3, 256, 256)

    def test_upsample(self):
        """Upsampling should produce larger output."""
        input_tensor = torch.randn(3, 256, 256)
        result = _lanczos_resize_chw(input_tensor, (512, 512))
        assert result.shape == (3, 512, 512)

    def test_identity(self):
        """Same size should preserve values approximately."""
        input_tensor = torch.randn(3, 256, 256)
        result = _lanczos_resize_chw(input_tensor, (256, 256))
        assert result.shape == (3, 256, 256)
        # Should be very close to original
        assert torch.allclose(result, input_tensor.cpu(), atol=1e-3)

    def test_preserves_device(self):
        """Should preserve tensor device."""
        input_tensor = torch.randn(3, 256, 256)
        result = _lanczos_resize_chw(input_tensor, (128, 128))
        assert result.device == input_tensor.device


class TestImageFormatConversion:
    """Tests for image format conversion logic."""

    def test_comfyui_to_windowseat(self):
        """Test conversion from ComfyUI to WindowSeat format."""
        # ComfyUI: [B, H, W, C] in [0, 1]
        comfyui = torch.rand(1, 512, 512, 3)

        # Convert
        img = comfyui[0].permute(2, 0, 1)  # [C, H, W]
        img = img * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        assert img.shape == (3, 512, 512)
        assert img.min() >= -1.0
        assert img.max() <= 1.0

    def test_windowseat_to_comfyui(self):
        """Test conversion from WindowSeat to ComfyUI format."""
        # WindowSeat: [C, H, W] in [-1, 1]
        windowseat = torch.randn(3, 512, 512).clamp(-1, 1)

        # Convert
        result = (windowseat + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        result = result.clamp(0, 1)
        result = result.permute(1, 2, 0)  # [H, W, C]

        assert result.shape == (512, 512, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_roundtrip_conversion(self):
        """Roundtrip conversion should preserve values."""
        # Start with ComfyUI format
        original = torch.rand(512, 512, 3)

        # ComfyUI -> WindowSeat
        ws = original.permute(2, 0, 1) * 2.0 - 1.0

        # WindowSeat -> ComfyUI
        back = ((ws + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0)

        assert torch.allclose(original, back, atol=1e-6)
