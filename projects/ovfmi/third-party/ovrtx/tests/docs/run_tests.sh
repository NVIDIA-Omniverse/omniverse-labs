#!/usr/bin/env bash
# Run all doc test suites (usd, python, c) and report results.
# Usage: ./run_tests.sh [pytest args...]
# Example: ./run_tests.sh -v --tb=short

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTEST_ARGS=("$@")

failed=()
passed=()

run_suite() {
    local name="$1"
    local dir="$SCRIPT_DIR/$name"

    echo "========================================"
    echo "Running: $name"
    echo "========================================"

    if (cd "$dir" && uv run pytest "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"); then
        passed+=("$name")
    else
        failed+=("$name")
    fi

    echo ""
}

run_suite usd
run_suite python

# C tests: build + run via dedicated script
echo "========================================"
echo "Running: c (CMake + GoogleTest)"
echo "========================================"

if "$SCRIPT_DIR/c/run_tests.sh"; then
    passed+=("c")
else
    failed+=("c")
fi

echo ""

echo "========================================"
echo "Results"
echo "========================================"

for suite in "${passed[@]}"; do
    echo "  PASSED: $suite"
done

for suite in "${failed[@]}"; do
    echo "  FAILED: $suite"
done

if [ ${#failed[@]} -gt 0 ]; then
    echo ""
    echo "${#failed[@]} suite(s) failed."
    exit 1
else
    echo ""
    echo "All suites passed."
    exit 0
fi
