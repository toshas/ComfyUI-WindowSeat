#!/bin/bash
# Run tests for ComfyUI-WindowSeat
#
# Usage:
#   ./run_tests.sh          # Run unit tests only
#   ./run_tests.sh --gpu    # Run all tests including GPU integration tests

set -e

# Change to script directory
cd "$(dirname "$0")"

# Run unit tests (no GPU needed)
echo "=== Running unit tests ==="
python -m pytest tests/test_core.py tests/test_nodes.py -v

# Run integration tests if --gpu flag is provided
if [[ "$1" == "--gpu" ]]; then
    echo "=== Running integration tests ==="
    python -m pytest tests/test_integration.py -v --gpu
fi

echo "=== All tests completed ==="
