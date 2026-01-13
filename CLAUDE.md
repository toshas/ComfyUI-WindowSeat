# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom node plugin for removing reflections from images using the WindowSeat model (based on Qwen-Image-Edit with LoRA fine-tuning). Downloads models from HuggingFace Hub on first use.

## Commands

### Setup Development Environment
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
pre-commit install  # Required: install git hooks
```

### Lint and Format
```bash
ruff check --fix .   # Lint and auto-fix
ruff format .        # Format code
ruff check . && ruff format --check .  # Check only (CI mode)
```

### Run Tests
```bash
# Unit tests only (no GPU required)
./run_tests.sh

# All tests including GPU integration tests
./run_tests.sh --gpu

# Or run pytest directly:
python -m pytest tests/test_core.py tests/test_nodes.py -v        # unit tests
python -m pytest tests/test_integration.py -v --gpu               # integration tests

# Single test class or function
python -m pytest tests/test_core.py::TestComputeTiles -v
python -m pytest tests/test_core.py::TestComputeTiles::test_large_image_multiple_tiles -v
```

## Architecture

### Core Components

- **windowseat_core.py** - Core inference logic:
  - `load_network()` - Loads VAE, transformer (NF4 quantized), LoRA adapter, and text embeddings
  - `process_image()` - Main processing pipeline: tiles image → encodes → runs flow step → decodes → stitches
  - `compute_tiles()` - Computes tile grid with overlap for large images
  - `stitch_tiles()` - Blends tiles using triangular window weighting

- **nodes.py** - ComfyUI node definitions:
  - `WindowSeatModelLoader` - Loads and caches model (class-level cache)
  - `WindowSeatReflectionRemoval` - Processes images with tiling support

### Data Flow

1. ComfyUI format: `[B, H, W, C]` in `[0, 1]` range
2. WindowSeat format: `[C, H, W]` in `[-1, 1]` range
3. Conversion happens in `WindowSeatReflectionRemoval.remove_reflections()`

### Tiling System

For high-res images, the system:
1. Computes optimal square tile grid respecting `max_tiles_w/h` and `min_overlap`
2. Scales up small images to meet `processing_resolution` (768px default)
3. Processes tiles in batches through VAE encode → transformer flow step → VAE decode
4. Stitches results with triangular window blending for smooth seams

### Model Components (from HuggingFace)

- Base: `Qwen/Qwen-Image-Edit-2509` (VAE + transformer)
- LoRA: `huawei-bayerlab/windowseat-reflection-removal-v1-0`
- Transformer uses NF4 quantization via bitsandbytes

## Testing

- `tests/test_core.py` - Unit tests for tile computation, stitching, resizing (no GPU)
- `tests/test_nodes.py` - Unit tests for ComfyUI node interfaces (no GPU)
- `tests/test_integration.py` - Full pipeline tests (requires `--gpu` flag)
- `tests/conftest.py` - Fixtures and pytest configuration for `--gpu` marker
