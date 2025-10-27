#!/usr/bin/env bash
# ============================================================
# SGemm Transform Runner
# Applies the SGemm transform (sgem.mlir) to the test payload
# ============================================================

set -e  # Stop on first error

# Define paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$ROOT_DIR/test"
OUT_DIR="$TEST_DIR/output"
TRANSFORM_FILE="$TEST_DIR/sgem.mlir"
PAYLOAD_FILE="$TEST_DIR/payload.mlir"
OUTPUT_FILE="$OUT_DIR/transformed.mlir"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Running SGemm transform..."
echo "Input payload : $PAYLOAD_FILE"
echo "Transform file: $TRANSFORM_FILE"
echo "Output file   : $OUTPUT_FILE"
echo "============================================================"

# Run transform-opt
sgemm-opt -transform="$TRANSFORM_FILE" "$PAYLOAD_FILE" > "$OUTPUT_FILE"

echo "Transform completed successfully!"
echo "Output written to $OUTPUT_FILE"

