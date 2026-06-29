#!/usr/bin/env bash
# Build and run the C doc tests.
# Usage: ./run_tests.sh [--prefix /path/to/ovrtx]
# If --prefix is not given, CMAKE_PREFIX_PATH must be set in the environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

PREFIX_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

CMAKE_ARGS=()
if [[ -n "$PREFIX_PATH" ]]; then
    CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$PREFIX_PATH")
fi

echo "=== Configuring ==="
cmake -B "$BUILD_DIR" -G Ninja -S "$SCRIPT_DIR" "${CMAKE_ARGS[@]}"

echo "=== Building ==="
cmake --build "$BUILD_DIR" --config Release

echo "=== Running tests ==="
cd "$BUILD_DIR"
ctest --output-on-failure -C Release
