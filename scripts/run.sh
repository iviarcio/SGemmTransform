#!/usr/bin/env bash
#
# Run script for mlir files using the LLVM MLIR CPU runner
# Make sure to set the LLVM_BIN_PATH variable to the LLVM build bin path
#
# MLIR lowering is done using mlir-opt

err() {
  echo "$*" >&2
  exit 1
}

# Check parameters
if [ $# -eq 0 ]; then
  err "Usage: ./run.sh <path-to-mlir-file>.mlir [-O]"
fi

readonly INPUT=$1
shift
OPTIMIZATION=""
while getopts 'O' flag; do
  case "${flag}" in
  O) OPTIMIZATION="-O3" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done
readonly OPTIMIZATION

# Library path
readonly LLVM_LIB_PATH="/home/marcio/work/llvm-project/build/lib"

# Check if input is a .mlir file
INPUT_BASE=$(basename "$INPUT" .mlir)
readonly INPUT_BASE
if [ "$INPUT_BASE" == "$INPUT" ]; then
  err "Input needs to be a .mlir file"
fi

# Create output file name
INPUT_DIR=$(dirname "$INPUT")
readonly INPUT_DIR
readonly OUTPUT="$INPUT_DIR"/"$INPUT_BASE".llvm.mlir

# Lower MLIR to LLVM
mlir-opt "$INPUT" \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-affine-loops --canonicalize --cse \
  --convert-vector-to-llvm='enable-x86vector' \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf \
  --normalize-memrefs --memref-expand --finalize-memref-to-llvm \
  --lower-affine --convert-func-to-llvm --convert-arith-to-llvm \
  --convert-cf-to-llvm --canonicalize --cse --symbol-dce \
  --llvm-legalize-for-export \
  -o "$OUTPUT"

# Add MLIR utils library
unameOut="$(uname -s)"
case "${unameOut}" in
Linux*)
  ext="so"
  ;;
Darwin*)
  ext="dylib"
  ;;
*)
  err "Unrecognized OS"
  ;;
esac

MLIR_RUNNER_UTILS=$(
  cat <<END
$LLVM_LIB_PATH/libmlir_c_runner_utils.$ext,$LLVM_LIB_PATH/libmlir_runner_utils.$ext
END
)

if [[ -n "${SCONV_EXTRA_LIB}" ]]; then
    MLIR_RUNNER_UTILS="$MLIR_RUNNER_UTILS,$SCONV_EXTRA_LIB"
fi
readonly MLIR_RUNNER_UTILS

# Run code
mlir-runner "$OUTPUT" -e main -entry-point-result=void \
  -shared-libs="$MLIR_RUNNER_UTILS" $OPTIMIZATION
