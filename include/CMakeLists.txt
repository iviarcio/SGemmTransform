# Tell Tablegen to use SGemm.td as input.
set(LLVM_TARGET_DEFINITIONS SGemm.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(SGemm.h.inc -gen-op-decls)
mlir_tablegen(SGemm.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(SGemmDialectIncGen)

# Don't forget to generate the documentation, this will produce a
# SGemmDialect.md under docs/Dialects
add_mlir_doc(SGemm SGemmDialect docs/Dialects/ -gen-op-doc)
