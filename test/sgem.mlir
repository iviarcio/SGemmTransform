module attributes {transform.with_named_sequence} {

  // Entry point for the transform interpreter.
  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op) {

    // Match all linalg.matmul ops inside the payload module/function.
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.matmul">

    // Apply the sliced GEMM transform Op (tiling/packing, lowered as linalg.generic).
    // Returns the transformed matmuls as linalg.generic plus loops handle.
    %res_list, %loops_list = transform.structured.sgemm %matmuls
      { mK_info = [8, 16] }
      : (!transform.op<"linalg.matmul">)
      -> (!transform.op<"linalg.generic">, !transform.any_op)

    // transform.foreach %res_list : !transform.op<"linalg.generic"> {
    //   // Insert additional transformations
    // }

    transform.yield
  }
}

