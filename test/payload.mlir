// Tipos de tensor
!a_t = tensor<2000x2000xf32>   // A: M=2000, K=2000
!b_t = tensor<2000x2000xf32>   // B: K=2000, N=2000
!c_t = tensor<2000x2000xf32>   // C: M=2000, N=2000

module {
  func.func @matmul(%A: !a_t, %B: !b_t, %Cinit: !c_t) -> !c_t {
    %C = linalg.matmul
      ins(%A, %B : !a_t, !b_t)
      outs(%Cinit : !c_t) -> !c_t
    return %C : !c_t
  }
}
