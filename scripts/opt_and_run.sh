#!/usr/bin/env bash
#
# Script to optimize and run MLIR files using a transform sequence.
# Uses opt.sh and run.sh scripts.

err() { echo "$*" >&2; exit 1; }

# Check parameters
if [ $# -lt 2 ]; then
  err "Usage: ./opt_and_run.sh <payload-ir>.mlir <transform-ir>.mlir [-O]"
fi
readonly PAYLOAD=$1
readonly TRANSFORM=$2
shift 2
OPTIMIZATION=""
while getopts 'O' flag; do
  case "${flag}" in
  O) OPTIMIZATION="-O" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done
readonly OPTIMIZATION

# Create output file name
INPUT_BASE=$(basename "$PAYLOAD" .mlir)
readonly INPUT_BASE

INPUT_DIR=$(dirname "$PAYLOAD")
readonly INPUT_DIR
readonly OUTPUT="$INPUT_DIR"/"$INPUT_BASE".opt.mlir

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
readonly SCRIPT

# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
readonly SCRIPTPATH

# Apply transform to MLIR payload
"$SCRIPTPATH"/opt.sh "$PAYLOAD" "$TRANSFORM"

# JIT compile and run optimized MLIR payload
"$SCRIPTPATH"/run.sh "$OUTPUT" "$OPTIMIZATION"
