//===-- SGemm.cpp - transform dialect Extension SGemm ------------------===//
//----------------------------------------------------------------------------//

#include "SGemm.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

#define GET_OP_CLASSES
#include "SGemm.cpp.inc"

/// Define the SGemm transform dialect. This uses the CRTP idiom to identify extensions.
class SGemm
    : public transform::TransformDialectExtension<SGemm> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SGemm)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void SGemm::init() {
  // As an transform extension dialect, we must declare all dependent dialects.
  // These dialects will be loaded along with the extension and, therefore,
  // along with the Transform dialect. The dependent dialects contain the
  // attributes or types used by transform operations.
  declareDependentDialect<linalg::LinalgDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  declareGeneratedDialect<affine::AffineDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<index::IndexDialect>();
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<tensor::TensorDialect>();
  declareGeneratedDialect<LLVM::LLVMDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "SGemm.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Packing helpers (row-panel for A, col-panel for B)
//===----------------------------------------------------------------------===//

/// Return the ExtractSliceOp that ultimately produces `v`, or null if none.
static tensor::ExtractSliceOp getSliceProducerOrNull(Value v) {
  Value cur = v;
  while (Operation *def = cur.getDefiningOp()) {
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def))
      return slice;
    // Skip through trivial wrappers.
    continue;
  }
  return nullptr;
}

/// Conservatively check whether `slice` is invariant in the current `loop`.
static bool isExtractSliceInvariantToLoop(tensor::ExtractSliceOp slice,
                                          scf::ForOp loop) {
  if (!loop) return true;
  auto *loopBody = loop.getBody();
  Value iv = loop.getInductionVar();
  for (Value v : slice->getOperands()) {
    if (!v) continue;
    if (v == iv) return false;
    Operation *def = v.getDefiningOp();
    if (!def) {
      // if it belongs to the loop body, be conservative.
      if (auto barg = dyn_cast<BlockArgument>(v))
        if (barg.getOwner() == loopBody) return false;
      continue;
    }
    if (loop->isAncestor(def)) return false;
  }
  return true;
}

/// Compute packed type for A: from [Mc, Kc] to [Mc/mr, Kc, mr].
static RankedTensorType
computeAPackedType(RankedTensorType aType, int64_t mr) {
  auto shape = aType.getShape();
  int64_t Mc = shape[0], Kc = shape[1];
  assert(Mc % mr == 0 && "Mc must be a multiple of mr for simple packing");
  SmallVector<int64_t, 3> newShape = {Mc / mr, Kc, mr};
  return RankedTensorType::get(newShape, aType.getElementType());
}

/// Compute packed type for B: from [Kc, Nc] to [Kc, Nc/nr, nr].
static RankedTensorType
computeBPackedType(RankedTensorType bType, int64_t nr) {
  auto shape = bType.getShape();
  int64_t Kc = shape[0], Nc = shape[1];
  assert(Nc % nr == 0 && "Nc must be a multiple of nr for simple packing");
  SmallVector<int64_t, 3> newShape = {Kc, Nc / nr, nr};
  return RankedTensorType::get(newShape, bType.getElementType());
}

/// Build affine maps for A-pack:  in:  (i,k)  -> out: (io, k, ii)
/// with io=floor(i/mr), ii=mod(i,mr)
static void buildAPackMaps(MLIRContext *ctx, int64_t mr,
                           AffineMap &inMap, AffineMap &outMap) {
  // dims: (i,k)
  AffineExpr i, k;
  bindDims(ctx, i, k);
  AffineExpr io = i.floorDiv(mr);
  AffineExpr ii = i % mr;
  inMap  = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{i, k}, ctx);
  outMap = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{io, k, ii}, ctx);
}

/// Build affine maps for B-pack:  in:  (k,j)  -> out: (k, jo, ji)
/// with jo=floor(j/nr), ji=mod(j,nr)
static void buildBPackMaps(MLIRContext *ctx, int64_t nr,
                           AffineMap &inMap, AffineMap &outMap) {
  // dims: (k,j)
  AffineExpr k, j;
  bindDims(ctx, k, j);
  AffineExpr jo = j.floorDiv(nr);
  AffineExpr ji = j % nr;
  inMap  = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{k, j}, ctx);
  outMap = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{k, jo, ji}, ctx);
}

/// Build A_pack at a precise insertion point. If `afterOp` != nullptr,
/// insert right AFTER it; otherwise insert BEFORE `beforeOp`.
/// in  : A tile type [Mc, Kc]
/// out : A_pack type [Mc/mr, Kc, mr]
static FailureOr<std::pair<Value, RankedTensorType>>
buildAPackAt(RewriterBase &rewriter, Location loc, Value A, int64_t mr,
             Operation *afterOp, Operation *beforeOp) {
  OpBuilder::InsertionGuard g(rewriter);

  if (afterOp) {
    rewriter.setInsertionPointAfter(afterOp);
  } else if (beforeOp) {
    rewriter.setInsertionPoint(beforeOp);
  } else {
    return failure();
  }

  auto aType = dyn_cast<RankedTensorType>(A.getType());
  if (!aType || aType.getRank() != 2) return failure();

  auto aPackTy = computeAPackedType(aType, mr);
  AffineMap aInMap, aOutMap;
  buildAPackMaps(rewriter.getContext(), mr, aInMap, aOutMap);

  Value aEmpty = rewriter.create<tensor::EmptyOp>(
      loc, aPackTy.getShape(), aPackTy.getElementType());

  auto aPack = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{aPackTy}, ValueRange{A}, ValueRange{aEmpty},
      ArrayRef<AffineMap>{aInMap, aOutMap},
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [&](OpBuilder &b, Location nloc, ValueRange args) {
        b.create<linalg::YieldOp>(nloc, args[0]);
      });
  aPack->setAttrs({{"APacking", rewriter.getUnitAttr()}});
  return std::make_pair(aPack.getResult(0), aPackTy);
}

// Build B_pack at the current insertion point.
// in  : B tile type [Kc, Nc]
// out : B_pack type [Kc, Nc/nr, nr]
static FailureOr<std::pair<Value, RankedTensorType>>
buildBPackAt(RewriterBase &rewriter, Location loc, Value B, int64_t nr) {
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!bType || bType.getRank() != 2) return failure();
  auto bPackTy = computeBPackedType(bType, nr);

  AffineMap bInMap, bOutMap;
  buildBPackMaps(rewriter.getContext(), nr, bInMap, bOutMap);

  Value bEmpty = rewriter.create<tensor::EmptyOp>(loc, bPackTy.getShape(),
                                                  bPackTy.getElementType());
  auto bPack = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{bPackTy}, ValueRange{B}, ValueRange{bEmpty},
      ArrayRef<AffineMap>{bInMap, bOutMap},
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [&](OpBuilder &b, Location nloc, ValueRange args) {
        b.create<linalg::YieldOp>(nloc, args[0]);
      });
  bPack->setAttrs({{"BPacking", rewriter.getUnitAttr()}});
  return std::make_pair(bPack.getResult(0), bPackTy);
}

/// Rewrite ukernel (linalg.generic) to read from A_pack / B_pack.
/// Keeps the same output (C tile) and body (mul+add), only remaps inputs.
static FailureOr<linalg::GenericOp>
rewriteUkernelToUsePackedAB(RewriterBase &rewriter, linalg::GenericOp ukernel,
                            Value aPack, Value bPack) {
  Location loc = ukernel.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Old operands
  Value oldA = ukernel.getDpsInputs()[0];
  Value oldB = ukernel.getDpsInputs()[1];
  Value oldC = ukernel.getDpsInits()[0];

  // Original maps (expect A(i,k), B(k,j), C(i,j))
  SmallVector<AffineMap> maps = llvm::to_vector(ukernel.getIndexingMapsArray());
  if (maps.size() != 3) return failure();

  // Build new maps: replace A/B maps to point into packed layouts.
  // We assume iterators are [i (parallel), k (reduction), j (parallel)] after outer+inner tiling.
  // Use 3-dim domain for the ukernel body referring to (i,k,j) as before.
  AffineExpr i, k, j;
  bindDims(ctx, i, k, j);

  // A_pack is indexed by (io, k, ii) with io=floor(i/mr), ii=mod(i,mr).
  // We don't have 'mr' here; infer from the aPack type: last dim is 'mr'.
  auto aPackTy = cast<RankedTensorType>(aPack.getType());
  int64_t mr = aPackTy.getShape().back();
  AffineExpr io = i.floorDiv(mr);
  AffineExpr ii = i % mr;

  // B_pack is indexed by (k, jo, ji) with jo=floor(j/nr), ji=mod(j,nr).
  auto bPackTy = cast<RankedTensorType>(bPack.getType());
  int64_t nr = bPackTy.getShape().back();
  AffineExpr jo = j.floorDiv(nr);
  AffineExpr ji = j % nr;

  AffineMap aPackMap = AffineMap::get(/*dims=*/3, /*symbols=*/0,
                                      ArrayRef<AffineExpr>{io, k, ii}, ctx);
  AffineMap bPackMap = AffineMap::get(/*dims=*/3, /*symbols=*/0,
                                      ArrayRef<AffineExpr>{k, jo, ji}, ctx);
  AffineMap cMap     = AffineMap::get(/*dims=*/3, /*symbols=*/0,
                                      ArrayRef<AffineExpr>{i, j}, ctx);

  // Create the iterator types
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iters = {parallel, reduction, parallel};

  // Clone as a new generic that reads from A_pack/B_pack.
  rewriter.setInsertionPoint(ukernel);
  auto newUkernel = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/TypeRange{oldC.getType()},
      /*inputs=*/ValueRange{aPack, bPack},
      /*outputs=*/ValueRange{oldC},
      /*indexingMaps=*/ArrayRef<AffineMap>{aPackMap, bPackMap, cMap},
      /*iteratorTypes=*/iters,
      [&](OpBuilder &b, Location nloc, ValueRange args) {
        // args: Aelt, Belt, Celt
        Value mul = createMul(nloc, args[0], args[1], b);
        Value sum = createAdd(nloc, mul, args[2], b);
        b.create<linalg::YieldOp>(nloc, sum);
      });

  // Replace and erase old ukernel.
  rewriter.replaceOp(ukernel, newUkernel->getResults());
  return newUkernel;
}

//===----------------------------------------------------------------------===//
// Tiling Helpers
//===----------------------------------------------------------------------===//

static inline int64_t roundDownToMultiple(int64_t x, int64_t m) {
  if (m <= 0) return x;
  return (x / m) * m;
}

/// Adaptive selection based on problem size and micro-kernel shape.
static GemmTileSizes chooseGemmTileSizes(int64_t M, int64_t N, int64_t K,
                                         int64_t mr, int64_t nr) {
  const int64_t targetMBlocks = 8;   // try 8 mr-blocks along M
  const int64_t targetNBlocks = 4;   // try 4 nr-blocks along N
  const int64_t KcTarget      = 128; // conservative default for fp32
  const int64_t KAlign        = 8;   // align reduction chunk

  // mr/nr at least as provided
  int64_t useMr = std::max<int64_t>(mr, 1);
  int64_t useNr = std::max<int64_t>(nr, 1);

  // Adaptive Mc/Nc and multiples of mr/nr, without exceeding M/N
  int64_t maxMBlocks = std::max<int64_t>(1, (M > 0 && useMr > 0) ? (M / useMr) : 1);
  int64_t maxNBlocks = std::max<int64_t>(1, (N > 0 && useNr > 0) ? (N / useNr) : 1);
  int64_t Mc = std::max<int64_t>(useMr, std::min<int64_t>(targetMBlocks, maxMBlocks) * useMr);
  int64_t Nc = std::max<int64_t>(useNr, std::min<int64_t>(targetNBlocks, maxNBlocks) * useNr);

  // Preferable Kc, limited by K e alined
  int64_t Kc = (K > 0) ? std::min<int64_t>(KcTarget, K) : KcTarget;
  Kc = roundDownToMultiple(Kc, KAlign);
  if (Kc <= 0) Kc = (K > 0) ? std::min<int64_t>(K, KAlign) : KAlign;

  return {Mc, Nc, Kc, useMr, useNr};
}

/// Apply 2-level tiling to a linalg op with iterators [i (M) : parallel, k (K) : reduction, j (N) : parallel].
/// Produces: outer tiling (Mc,Kc,Nc) with interchange {0,2,1} (loops: i, j, k),
/// then, apply inner tiling (mr,0,nr) on the inner tiled op.
static LogicalResult
applyTileToGemm(RewriterBase &rewriter, Operation *transformOp, Operation *target,
                const mKInfo &mK, const ArchInfo &arch,
                SmallVector<Operation*, 6> &outResults) {

  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  // Infer problem sizes (M, N, K) from the target op types.
  // For linalg.matmul: A:[M,K], B:[K,N], C:[M,N].
  int64_t M = -1, N = -1, K = -1;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(target)) {
    auto A = linalgOp.getDpsInputs()[0];
    auto B = linalgOp.getDpsInputs()[1];
    auto C = linalgOp.getDpsInits()[0];

    if (auto aTy = dyn_cast<RankedTensorType>(A.getType())) {
      if (aTy.getRank() == 2) {
        // A:[M,K]
        M = aTy.getShape()[0];
        K = aTy.getShape()[1];
      }
    }
    if (auto bTy = dyn_cast<RankedTensorType>(B.getType())) {
      if (bTy.getRank() == 2) {
        // B:[K,N]
        if (K < 0) K = bTy.getShape()[0];
        N = bTy.getShape()[1];
      }
    }
    if (auto cTy = dyn_cast<RankedTensorType>(C.getType())) {
      if (cTy.getRank() == 2) {
        // C:[M,N]
        if (M < 0) M = cTy.getShape()[0];
        if (N < 0) N = cTy.getShape()[1];
      }
    }
  }

  // Fall back gracefully if any dim is dynamic/unknown.
  int64_t mr = mK.nrows;
  int64_t nr = mK.ncols;
  GemmTileSizes ts = chooseGemmTileSizes(M, N, K, mr, nr);

  // Outer (Mc,Kc,Nc) then inner (mr,0,nr) as before
  SmallVector<int64_t, 3> outerTileSz = {ts.Mc, ts.Kc, ts.Nc};
  SmallVector<OpFoldResult> outerTileOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), outerTileSz);
  SmallVector<int64_t, 3> outerInterchange = {0, 2, 1};

  scf::SCFTilingOptions outerOpts;
  outerOpts.setTileSizes(outerTileOfr).setInterchange(outerInterchange);
  outerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> outerRes =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, outerOpts);
  if (failed(outerRes))
    return transformOp->emitError("First level tiling for GEMM failed.");

  // Outer replace: use loop results when loops exist.
  if (!outerRes->loops.empty()) {
    // The scf.for returns the assembled tensor via yield -> results().
    rewriter.replaceOp(tilingInterfaceOp, outerRes->loops.front()->getResults());
  } else if (!outerRes->tiledOps.empty()) {
    // Degenerate case (no loops): fall back to the tiled op result.
    rewriter.replaceOp(tilingInterfaceOp, outerRes->tiledOps.front()->getResults());
  } else {
    // No tiled op produced: a safe fallback, just erase.
    rewriter.eraseOp(tilingInterfaceOp);
  }

  // Inner level: (mr, nr) on inner tiled op.
  Operation *innerOp = outerRes->tiledOps.front();

  SmallVector<int64_t, 3> innerTileSz = {ts.mr, /*K*/ 0, ts.nr};
  SmallVector<OpFoldResult> innerTileOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), innerTileSz);

  // Keep inner order (i, j) — reduction K stays as-is (no extra tiling).
  SmallVector<int64_t, 2> innerInterchange = {0, 1};

  auto innerTilingInterfaceOp = dyn_cast<TilingInterface>(innerOp);
  if (!innerTilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported (inner).");

  scf::SCFTilingOptions innerOpts;
  innerOpts.setTileSizes(innerTileOfr).setInterchange(innerInterchange);
  innerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(innerOp);
  FailureOr<scf::SCFTilingResult> innerRes =
      scf::tileUsingSCF(rewriter, innerTilingInterfaceOp, innerOpts);
  if (failed(innerRes))
    return transformOp->emitError("Second level tiling for GEMM failed.");

  // Inner replace: same rule as outer.
  if (!innerRes->loops.empty()) {
    rewriter.replaceOp(innerTilingInterfaceOp, innerRes->loops.front()->getResults());
  } else if (!innerRes->tiledOps.empty()) {
    rewriter.replaceOp(innerTilingInterfaceOp, innerRes->tiledOps.front()->getResults());
  } else {
    rewriter.eraseOp(innerTilingInterfaceOp);
  }

  // Report back the tiled inner op + all loops (outer + inner)
  tiledOps.push_back(innerRes->tiledOps.front());
  for (Operation *loop : outerRes->loops) loopOps.push_back(loop);
  for (Operation *loop : innerRes->loops) loopOps.push_back(loop);

  // Collect only real ops: [tiledInnerOp, loop...];
  outResults.clear();
  if (!tiledOps.empty() && tiledOps.front())
    outResults.push_back(tiledOps.front());
  for (Operation *L : loopOps)
    if (L) outResults.push_back(L);

  return success();
}

/// This function generalize a named linalg.matmul into a
/// canonical linalg.generic with A(i,k), B(k,j), C(i,j) and
/// iterator_types = ["parallel", "reduction", "parallel"].
/// Returns the created generic op or failure.
FailureOr<linalg::GenericOp>
generalizeMatmulToGeneric(RewriterBase &rewriter, linalg::MatmulOp matmul) {

  MLIRContext *ctx = rewriter.getContext();
  Location loc = matmul.getLoc();

  // Get DPS interfaces (inputs/inits).
  // Preconditions: ranks must be 2x2-> 2
  SmallVector<Value> inputs = matmul.getDpsInputs();
  ValueRange inits = matmul.getDpsInits();
  if (inputs.size() != 2 || inits.size() != 1)
    return failure();

  Value A = inputs[0];
  Value B = inputs[1];
  Value Cinit = inits[0];

  // Sanity-check types.
  auto aType = dyn_cast<ShapedType>(A.getType());
  auto bType = dyn_cast<ShapedType>(B.getType());
  auto cType = dyn_cast<ShapedType>(Cinit.getType());
  if (!aType || !bType || !cType || aType.getRank() != 2 ||
      bType.getRank() != 2 || cType.getRank() != 2)
    return failure();

  // Build indexing maps: A(i,k), B(k,j), C(i,j).
  AffineExpr i, k, j;
  bindDims(ctx, i, k, j);

  AffineMap aMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                  ArrayRef<AffineExpr>{i, k}, ctx);
  AffineMap bMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                  ArrayRef<AffineExpr>{k, j}, ctx);
  AffineMap cMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                  ArrayRef<AffineExpr>{i, j}, ctx);

  // Create the iterator types
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iters = {parallel, reduction, parallel};

  // Create the linalg.generic and inline the multiply-accumulate.
  rewriter.setInsertionPoint(matmul);
  auto generic = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/TypeRange{cType},
      /*inputs=*/ValueRange{A, B},
      /*outputs=*/ValueRange{Cinit},
      /*indexingMaps=*/ArrayRef<AffineMap>{aMap, bMap, cMap},
      /*iteratorTypes=*/iters,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange blockArgs) {
        // blockArgs: %a, %b, %acc
        Value mul = createMul(nestedLoc, blockArgs[0], blockArgs[1], nested);
        Value sum = createAdd(nestedLoc, mul, blockArgs[2], nested);
        nested.create<linalg::YieldOp>(nestedLoc, sum);
      });

  // Replace the named op with the generic result.
  rewriter.replaceOp(matmul, generic->getResults());
  return generic;
}

///
/// Implementation of SGemm::apply transform dialect operation.
///
DiagnosedSilenceableFailure
transform::SGemmOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Initialize the default values of mKInfo & ArchInfo.
  // It's dependent of the target machine.
  mKInfo mK = {8, 16, 128};
  ArchInfo arch = {
      (uint32_t)(32768),
      (uint32_t)(1048576),
      (uint32_t)(4194304)
  };

  // Get the optional arguments
  auto mKInfoAttr = getMKInfo();
  auto archInfoAttr = getArchInfo();

  // If `mKInfoAttr` was provided, use the given values
  if (mKInfoAttr) {
    SmallVector<int64_t, 4> mKValues;
    for (auto attr : mKInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        mKValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: mKInfoAttr contains non-integer values!\n";
      }
    }
    if (mKValues.size() >= 2) {
      mK.nrows = mKValues[0];
      mK.ncols = mKValues[1];
      mK.noutput = mK.nrows * mK.ncols;
    } else {
      return emitSilenceableError() << "Error: mKInfoAttr does not contain enough values!\n";
    }
  }

  // If `archInfoAttr` was provided, use the given values
  if (archInfoAttr) {
    SmallVector<int64_t, 3> archValues;
    for (auto attr : archInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        archValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: archInfoAttr contains non-integer values!\n";
      }
    }
    if (archValues.size() >=3) {
      arch.l1_size = (uint32_t)(archValues[0]);
      arch.l2_size = (uint32_t)(archValues[1]);
      arch.l3_size = (uint32_t)(archValues[2]);
    } else {
      return emitSilenceableError() << "Error: archInfoAttr does not contain enough values!\n";
    }
  }

  // temporary variables to store all sgemm transformation
  SmallVector<Operation*> tempResultGemms;
  SmallVector<SmallVector<Operation*, 5>> tempResultLoops;

  // Get context and gemmOps
  MLIRContext *context = rewriter.getContext();
  auto targetOps = state.getPayloadOps(getTarget());

  for (Operation *t : targetOps) {
    if (auto mm = dyn_cast<linalg::MatmulOp>(t)) {
      auto guarded = transform::TransformRewriter::InsertionGuard(rewriter);
      rewriter.setInsertionPoint(mm);
      auto res = generalizeMatmulToGeneric(rewriter, mm);
      if (failed(res))
        return emitSilenceableError() << "failed to generalize linalg.matmul";

      SmallVector<Operation*, 6> localResults;

      linalg::GenericOp genericOp = *res;
      if (failed(applyTileToGemm(rewriter, getOperation(), genericOp.getOperation(), mK, arch, localResults)))
        return emitSilenceableError() << "Failed to apply tiling.";

      // localResults[0] contain the tiled uKernel (guard against missing/nullable)
      auto ukernel = dyn_cast<linalg::GenericOp>(localResults[0]);
      if (ukernel) {
        int64_t mr = mK.nrows;
        int64_t nr = mK.ncols;

        Value A = ukernel.getDpsInputs()[0]; // slice local [Mc_local, K_local]
        Value B = ukernel.getDpsInputs()[1]; // slice local [K_local, N_local]

        // If the local width of this ukernel (Nc_local) is not a multiple of nr,
        // skip packing on N-tail (simple fallback)
        if (auto bTy = dyn_cast<RankedTensorType>(B.getType())) {
          if (bTy.getRank() == 2) {
            int64_t N_local = bTy.getShape()[1];
            if (N_local == ShapedType::kDynamic || nr == 0 || (N_local % nr) != 0) {
              // keep ukernel as-is (no A/B pack); optional debug attr:
              ukernel->setAttr("RemainderN",
                rewriter.getI64IntegerAttr(N_local == ShapedType::kDynamic ? -1 : N_local));
              // continue to next ukernel
              continue; // (skip hoisting/packing)
            }
          }
        }

        // Try to hoist: move A's slice above the inner loop if invariant
        Operation *afterA = nullptr;
        Operation *beforeA = ukernel.getOperation();
        // Get the innermost enclosing For
        if (auto innerFor = dyn_cast<scf::ForOp>(localResults[5])) {
          // Find the slice that produces A
          if (auto aSlice = getSliceProducerOrNull(A)) {
            if (isExtractSliceInvariantToLoop(aSlice, innerFor)) {
              // Move the slice that feeds A outside the inner loop.
              aSlice->moveBefore(innerFor);
              afterA = aSlice.getOperation(); // Insert A_pack right after this slice
            }
          }
        }

        // Build A_pack from the local slices A
        auto loc = ukernel.getLoc();
        auto aPackOr = buildAPackAt(rewriter, loc, A, mr, afterA, beforeA);
        if (failed(aPackOr))
          return emitSilenceableError() << "A_pack failed.\n";
        Value A_pack = aPackOr->first;

        // Build B_pack from the local slices B
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(ukernel);
        auto bPackOr = buildBPackAt(rewriter, loc, B, nr);
        if (failed(bPackOr))
          return emitSilenceableError() << "B_pack failed.\n";
        Value B_pack = bPackOr->first;

        // If both packs exist, rewrite the ukernel to consume them.
        if (A_pack && B_pack) {
          auto newUkernelOr =
              rewriteUkernelToUsePackedAB(rewriter, ukernel, A_pack, B_pack);
          // Update localResults[0] with ukernel
          if (succeeded(newUkernelOr)) {
            linalg::GenericOp newUkernel = *newUkernelOr;
            newUkernel->setAttrs({{"microkernel", rewriter.getUnitAttr()}});
            if (!localResults.empty())
              localResults[0] = newUkernel.getOperation();
          }
        }
      }

      // Store tiled uKernel (localResults[0])
      if (!localResults.empty() && localResults[0])
        tempResultGemms.push_back(localResults[0]);

      // Following, store the generated loops
      SmallVector<Operation*, 5> loopSet;
      for (size_t i = 1; i < localResults.size(); ++i)
        if (localResults[i]) loopSet.push_back(localResults[i]);
      tempResultLoops.push_back(loopSet);

    } // if matmulOp
  } // for targetOps

  // Flatten tempResultLoops
  SmallVector<Operation*> flatResultLoops;
  for (const auto &loopSet : tempResultLoops) {
    for (Operation *L : loopSet)
      if (L) flatResultLoops.push_back(L);
  }

  // Also filter nulls from tempResultGemms, just in case.
  {
    SmallVector<Operation*> filtered;
    filtered.reserve(tempResultGemms.size());
    for (Operation *op : tempResultGemms)
      if (op) filtered.push_back(op);
    tempResultGemms.swap(filtered);
  }

  // Store results properly in TransformResults
  results.set(getOperation()->getOpResult(0), tempResultGemms);
  results.set(getOperation()->getOpResult(1), flatResultLoops);

  return DiagnosedSilenceableFailure::success();

} // SGemmOp::apply

void transform::SGemmOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult
transform::SGemmOp::verify() {
  // All necessary checks are done in the Apply
  return success();
}

void registerSGemm(mlir::DialectRegistry &registry) {
  registry.addExtensions<SGemm>();
}
