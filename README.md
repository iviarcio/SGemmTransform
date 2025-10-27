# SGemm Transform

## Overview

`SGemm` is an extension of the MLIR *Transform Dialect* that detects `linalg.matmul` operations modeling GEMM and applies a sequence of transformations to produce a tiled version with packing and the possibility of replacing the micro-kernel with highly optimized calls. The main goal is to prepare the payload (IR at the linalg level) for later lowering and substitution of the micro-kernel by external or target-specific routines. The registration of the extension and the entry point of the transform are defined in the extension code.

## Repository contents (main files)

* `SGemm.td` — Defines the Transform dialect extension operations used by SGemm.
* `SGemm.h` — Declarations, utilities, structs, and debug macros (for example, `DEBUG_TYPE` and `DBGS()`), useful typedefs (e.g., `GemmTileSizes`, `mKInfo`).
* `SGemm.cpp` — Implementation of the transform (main logic: matmul generalization, padding, tiling, packing, and micro-kernel retargeting).
* `payload.mlir` — Test payload (input example).
* `sgem.mlir` — Transform file configuring `SGemm` (parameters mK, arch, etc.).

## How to invoke (example)

To apply the transformation (still without lowering to LLVM), use:

```bash
sgemm-opt -transform=sgem.mlir payload.mlir
```

This command loads the transform dialect and executes the operations defined in `sgem.mlir` on the payload `payload.mlir`. The pass searches for `linalg.matmul` operations modeling GEMM and applies the transformation.

## Internal workflow — step by step (high level)

1. **Generalize `linalg.matmul` to `linalg.generic`**
   The transform begins by generalizing `matmul` into a `linalg.generic` operation to have a uniform representation on which to apply tiling and packing.

2. **Compute tile/micro-kernel parameters (`GemmTileSizes`)**
   The pass computes outer dimensions (Mc, Kc, Nc) and inner ones (mr, nr) based on `mK` (e.g., 8x16) and architecture information. These parameters guide both padding and tiling/packing.

3. **Prepare and apply *padding* (if necessary)**

   * The transform checks whether the shapes of `A`, `B`, and `C` are already multiples of the desired block sizes. The function `needsPaddingToMultiples` returns `None` (no padding needed), `Needed` (static padding required), or `Maybe` (some dynamic dimension — keep padding path active).
   * If at least one tensor requires padding, the pass creates `tensor.pad` (or equivalent ops) immediately before the original GEMM operation and returns the possibly padded tensors for the subsequent pipeline. The verification and creation of the `pad` ops occur in `preparePaddingForGemmLike` and `padToMultiples`.

   Notes:

   * If all three (A, B, C) are statically multiples of the block sizes, **no** `pad` will be emitted, and the IR remains “pristine” (no padding prelude).
   * When padding is created, the pass **clones** the original `linalg.generic` so that the padded version replaces the original. This substitution happens before tiling.

4. **Apply tiling**
   After the padding/clone step, the pass applies a two-level tiling scheme (outer + micro-kernel), producing the tiled version where one of the local results represents the uKernel (micro-kernel) that will be packed/substituted. Tiling uses the previously computed `GemmTileSizes`.

5. **Packing and micro-kernel retargeting**
   The result of the tiling step is passed to routines that:

   * build packed buffers for A and B (and possibly multi-packing depending on schedule),
   * adjust affine maps and indexing so that the micro-kernel sees memory in the expected layout,
   * and finally retarget the micro-kernel call (for example, to sgemm/OpenBLAS calls or specific routines).
     The transform infrastructure allows registering operations that generate `linalg.generic` with affine maps and the necessary packing.

## *Padding* behavior — practical details

* **Which dimensions are checked for multiples**

  * A — must be a multiple of `(mr, Kc)` (A has shape `[M, K]`).
  * B — must be a multiple of `(Kc, nr)` (B has shape `[K, N]`).
  * C — must be a multiple of `(mr, nr)` (C has shape `[M, N]`).
    These checks are implemented in `needsPaddingToMultiples`.

* **What happens when a dimension is dynamic?**

  * When a dimension is dynamic, the function returns `PadNeed::Maybe` to keep the padding path enabled — this ensures that the generated IR includes the logic to decide padding at runtime.

* **Where is `tensor.pad` inserted?**

  * The `pad` (when necessary) is inserted immediately before the GEMM operation (insertion point is `gemmLike`), so the rest of the flow (clone of `generic`, tiling, packing) operates on the padded tensors.

## Parameters and heuristics (where to tune)

* The micro-kernel schedule (`mK`) and dimensions `mr/nr` are configurable (set in `sgem.mlir`, which you use for invocation). These values drive the computation of `GemmTileSizes` and are used to choose `Kc`, `Mc`, `Nc` and dimension the packing.
* Architecture information (`ArchInfo`) can be used to adjust `Kc`/`Mc` to fit within L1/L2/L3 caches. See the structs and locations where `arch` is consulted during initialization and tile computation.

## Debug and logging

* You can enable debug output for the pass with:

  ```bash
  sgemm-opt ... -debug-only=SGemm
  ```

  The header defines `DEBUG_TYPE "sgemm"` and helper macros; use `DBGS()` in the code to print debug messages.

## Testing and payload

* The test payload (`payload.mlir`) validates that:

  * The generalization of `matmul` to `generic` works,
  * the padding path is triggered when required,
  * tiling and packing produce the expected ops (look for cloned `linalg.generic`, `tensor.pad`, and resulting packing ops).

* Recommended testing flow:

  1. `sgemm-opt -transform=sgem.mlir payload.mlir > transformed.mlir`
  2. Inspect `transformed.mlir` for `tensor.pad`, cloned `linalg.generic`, packing ops, and the resulting micro-kernel.
  3. Optionally, apply the lowering passes (bufferize, convert-linalg-to-affine-loops, etc.) to validate full lowering and execute with `mlir-runner` (for result comparison).

## Known limitations & TODOs

* **Lowering of the micro-kernel to external calls** — currently, the transformation prepares the micro-kernel. Still, full substitution with OpenBLAS/sgemm calls may require additional passes (bufferization, memref → LLVM conversion, and explicit call insertion).

* **Pad vs. split** — in some scenarios, it may be preferable to perform *split* (handle remainders with smaller blocks) instead of padding; SGemm currently prioritizes padding and maintains `Maybe` paths for dynamic cases.
