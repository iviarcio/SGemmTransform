#!/usr/bin/env bash
#
# Script to apply transformations to MLIR files
# Make sure to set the LLVM_BIN_PATH variable to the LLVM build bin path

err() {
  echo "$*" >&2
  exit 1
}

# Check parameters
if [ $# -lt 2 ]; then
  err "Usage: ./opt.sh <payload-ir>.mlir <transform-ir>.mlir [--no-file-output]"
fi

# Check for optional flag
NO_OUTPUT=0
if [[ "$3" == "--no-file-output" ]]; then
  NO_OUTPUT=1
fi

# Create output file name
INPUT_BASE=$(basename "$1" .mlir)
readonly INPUT_BASE

TRANSFORM_BASE=$(basename "$2" .mlir)
readonly TRANSFORM_BASE

# Check if input and transform are .mlir files
if [[ "$INPUT_BASE" == "$1" || "$TRANSFORM_BASE" == "$2" ]]; then
  err "Files needs to be .mlir"
fi

# Create output file name
INPUT_DIR=$(dirname "$1")
readonly INPUT_DIR
readonly OUTPUT="$INPUT_DIR"/"$INPUT_BASE".opt.mlir

# Optimize MLIR
if [[ $NO_OUTPUT -eq 1 ]]; then
  sgemm-opt -transform="$2" "$1"
else
  sgemm-opt -transform="$2" "$1" >"$OUTPUT"
fi
