"""Pytest fixtures for WindowSeat tests."""

import pytest
import torch


@pytest.fixture
def small_image_chw():
    """Create a small test image [C, H, W] in [-1, 1] range."""
    return torch.randn(3, 256, 256)


@pytest.fixture
def medium_image_chw():
    """Create a medium test image [C, H, W] in [-1, 1] range."""
    return torch.randn(3, 768, 768)


@pytest.fixture
def large_image_chw():
    """Create a large test image [C, H, W] in [-1, 1] range."""
    return torch.randn(3, 1536, 2048)


@pytest.fixture
def comfyui_batch():
    """Create a ComfyUI-format batch [B, H, W, C] in [0, 1] range."""
    return torch.rand(2, 512, 512, 3)


@pytest.fixture
def sample_tile_data():
    """Create sample tile data for stitching tests."""
    tiles = [
        (torch.randn(1, 3, 256, 256), (0, 0, 256, 256)),
        (torch.randn(1, 3, 256, 256), (192, 0, 448, 256)),
        (torch.randn(1, 3, 256, 256), (0, 192, 256, 448)),
        (torch.randn(1, 3, 256, 256), (192, 192, 448, 448)),
    ]
    return tiles, 448, 448


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU integration tests",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless --gpu flag is provided."""
    if config.getoption("--gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
