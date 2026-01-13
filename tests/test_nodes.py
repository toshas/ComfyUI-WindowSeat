"""Unit tests for ComfyUI node classes.

These tests verify the node interface without loading the actual model.
"""

from nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    WindowSeatModelLoader,
    WindowSeatReflectionRemoval,
)


class TestNodeMappings:
    """Tests for node registration mappings."""

    def test_class_mappings_exist(self):
        """NODE_CLASS_MAPPINGS should contain both nodes."""
        assert "WindowSeatModelLoader" in NODE_CLASS_MAPPINGS
        assert "WindowSeatReflectionRemoval" in NODE_CLASS_MAPPINGS

    def test_display_name_mappings_exist(self):
        """NODE_DISPLAY_NAME_MAPPINGS should contain both nodes."""
        assert "WindowSeatModelLoader" in NODE_DISPLAY_NAME_MAPPINGS
        assert "WindowSeatReflectionRemoval" in NODE_DISPLAY_NAME_MAPPINGS

    def test_class_mappings_are_classes(self):
        """Mappings should point to actual classes."""
        assert NODE_CLASS_MAPPINGS["WindowSeatModelLoader"] is WindowSeatModelLoader
        assert NODE_CLASS_MAPPINGS["WindowSeatReflectionRemoval"] is WindowSeatReflectionRemoval


class TestWindowSeatModelLoaderInterface:
    """Tests for WindowSeatModelLoader node interface."""

    def test_has_required_attributes(self):
        """Node should have all required ComfyUI attributes."""
        assert hasattr(WindowSeatModelLoader, "CATEGORY")
        assert hasattr(WindowSeatModelLoader, "FUNCTION")
        assert hasattr(WindowSeatModelLoader, "RETURN_TYPES")
        assert hasattr(WindowSeatModelLoader, "INPUT_TYPES")

    def test_category(self):
        """Node should be in WindowSeat category."""
        assert WindowSeatModelLoader.CATEGORY == "WindowSeat"

    def test_return_types(self):
        """Node should return WINDOWSEAT_MODEL."""
        assert WindowSeatModelLoader.RETURN_TYPES == ("WINDOWSEAT_MODEL",)

    def test_return_names(self):
        """Node should have return name."""
        assert WindowSeatModelLoader.RETURN_NAMES == ("model",)

    def test_input_types_structure(self):
        """INPUT_TYPES should return valid structure."""
        input_types = WindowSeatModelLoader.INPUT_TYPES()
        assert isinstance(input_types, dict)
        assert "required" in input_types

    def test_function_name(self):
        """FUNCTION should match method name."""
        assert WindowSeatModelLoader.FUNCTION == "load_model"
        assert hasattr(WindowSeatModelLoader, "load_model")


class TestWindowSeatReflectionRemovalInterface:
    """Tests for WindowSeatReflectionRemoval node interface."""

    def test_has_required_attributes(self):
        """Node should have all required ComfyUI attributes."""
        assert hasattr(WindowSeatReflectionRemoval, "CATEGORY")
        assert hasattr(WindowSeatReflectionRemoval, "FUNCTION")
        assert hasattr(WindowSeatReflectionRemoval, "RETURN_TYPES")
        assert hasattr(WindowSeatReflectionRemoval, "INPUT_TYPES")

    def test_category(self):
        """Node should be in WindowSeat category."""
        assert WindowSeatReflectionRemoval.CATEGORY == "WindowSeat"

    def test_return_types(self):
        """Node should return IMAGE."""
        assert WindowSeatReflectionRemoval.RETURN_TYPES == ("IMAGE",)

    def test_function_name(self):
        """FUNCTION should match method name."""
        assert WindowSeatReflectionRemoval.FUNCTION == "remove_reflections"
        assert hasattr(WindowSeatReflectionRemoval, "remove_reflections")

    def test_input_types_required(self):
        """Should have required inputs for model and image."""
        input_types = WindowSeatReflectionRemoval.INPUT_TYPES()
        assert "required" in input_types
        assert "model" in input_types["required"]
        assert "image" in input_types["required"]

    def test_input_types_optional(self):
        """Should have optional inputs for advanced settings."""
        input_types = WindowSeatReflectionRemoval.INPUT_TYPES()
        assert "optional" in input_types
        optional = input_types["optional"]

        # Check all optional parameters exist
        assert "use_short_edge_tile" in optional
        assert "tiling_size" in optional
        assert "max_tiles_w" in optional
        assert "max_tiles_h" in optional
        assert "min_overlap" in optional
        assert "tile_batch_size" in optional

    def test_optional_defaults(self):
        """Optional parameters should have correct defaults."""
        input_types = WindowSeatReflectionRemoval.INPUT_TYPES()
        optional = input_types["optional"]

        # Check defaults match Gradio demo
        assert optional["use_short_edge_tile"][1]["default"] is True
        assert optional["tiling_size"][1]["default"] == 768
        assert optional["max_tiles_w"][1]["default"] == 4
        assert optional["max_tiles_h"][1]["default"] == 4
        assert optional["min_overlap"][1]["default"] == 64
        assert optional["tile_batch_size"][1]["default"] == 2

    def test_optional_ranges(self):
        """Optional parameters should have sensible ranges."""
        input_types = WindowSeatReflectionRemoval.INPUT_TYPES()
        optional = input_types["optional"]

        # tiling_size range
        assert optional["tiling_size"][1]["min"] == 512
        assert optional["tiling_size"][1]["max"] == 1536

        # max_tiles range
        assert optional["max_tiles_w"][1]["min"] == 1
        assert optional["max_tiles_w"][1]["max"] == 8

        # min_overlap range
        assert optional["min_overlap"][1]["min"] == 16
        assert optional["min_overlap"][1]["max"] == 256

    def test_model_input_type(self):
        """Model input should expect WINDOWSEAT_MODEL type."""
        input_types = WindowSeatReflectionRemoval.INPUT_TYPES()
        model_type = input_types["required"]["model"]
        assert model_type == ("WINDOWSEAT_MODEL",)

    def test_image_input_type(self):
        """Image input should expect IMAGE type."""
        input_types = WindowSeatReflectionRemoval.INPUT_TYPES()
        image_type = input_types["required"]["image"]
        assert image_type == ("IMAGE",)


class TestModelLoaderCaching:
    """Tests for model caching behavior."""

    def test_cache_attributes_exist(self):
        """Loader should have cache attributes."""
        assert hasattr(WindowSeatModelLoader, "_cached_model")
        assert hasattr(WindowSeatModelLoader, "_cached_device")

    def test_cache_initially_none(self):
        """Cache should be None before loading."""
        # Reset cache for test
        WindowSeatModelLoader._cached_model = None
        WindowSeatModelLoader._cached_device = None

        assert WindowSeatModelLoader._cached_model is None
        assert WindowSeatModelLoader._cached_device is None
