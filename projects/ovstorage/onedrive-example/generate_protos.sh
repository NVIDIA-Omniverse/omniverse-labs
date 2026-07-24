#!/bin/bash
# Generate Python gRPC stubs from .proto files.
#
# This script compiles all .proto files under proto/ into Python _pb2.py and
# _pb2_grpc.py stubs under omni_onedrive_service/src/nvidia/...
#
# Prerequisites:
#   pip install grpcio-tools
#
# Usage:
#   ./generate_protos.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/proto"
OUTPUT_DIR="$SCRIPT_DIR/omni_onedrive_service/src"

# Find the protoc plugin from grpcio-tools
if ! python3 -m grpc_tools.protoc --help > /dev/null 2>&1; then
    echo "Error: grpcio-tools is not installed. Install it with:"
    echo "  pip install grpcio-tools"
    exit 1
fi

# Collect all .proto files
PROTO_FILES=$(find "$PROTO_DIR" -name "*.proto" | sort)

if [ -z "$PROTO_FILES" ]; then
    echo "No .proto files found in $PROTO_DIR"
    exit 1
fi

echo "Found proto files:"
for f in $PROTO_FILES; do
    echo "  ${f#$SCRIPT_DIR/}"
done
echo ""

echo "Generating stubs into ${OUTPUT_DIR#$SCRIPT_DIR/}/ ..."

python3 -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    --pyi_out="$OUTPUT_DIR" \
    $PROTO_FILES

# Ensure __init__.py files exist in all generated package directories
find "$OUTPUT_DIR/nvidia" -type d -exec touch {}/__init__.py \;

echo ""
echo "Proto generation complete."
