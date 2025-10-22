#!/usr/bin/env bash
# ============================================================
# SGemm Transform Checker
# Compares the produced transformed.mlir with the expected one
# ============================================================

set -e  # Stop on first error

# Define paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$ROOT_DIR/test"
OUT_DIR="$TEST_DIR/output"
GENERATED="$OUT_DIR/transformed.mlir"
EXPECTED="$TEST_DIR/transformed_expected.mlir"

# Check existence of required files
if [ ! -f "$GENERATED" ]; then
  echo "Error: Generated file not found: $GENERATED"
  echo "Run scripts/run_transform.sh first."
  exit 1
fi

if [ ! -f "$EXPECTED" ]; then
  echo "Error: Expected reference file not found: $EXPECTED"
  exit 1
fi

echo "============================================================"
echo "Checking transformed MLIR output..."
echo "Generated: $GENERATED"
echo "Expected : $EXPECTED"
echo "============================================================"

# Perform comparison (ignoring whitespace differences)
if diff -q -B -w "$GENERATED" "$EXPECTED" > /dev/null; then
  echo "✅ SGemm transform output matches the expected result!"
else
  echo "❌ Differences detected between generated and expected output."
  echo "Use 'diff -u $EXPECTED $GENERATED' to inspect differences."
  exit 1
fi

