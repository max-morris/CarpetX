#ifndef CARPETX_TESTREAL2_REQUIRES_REAL2_HXX
#define CARPETX_TESTREAL2_REQUIRES_REAL2_HXX

// TestReal2 is the CCTK_REAL2 (binary16) layer of the mixed-precision test
// suite, so every one of its translation units declares or manipulates
// CCTK_REAL2 grid variables, parameters, or aliased-function arguments. It
// therefore cannot be built at all on a configuration without CCTK_REAL2
// support, and the way it fails there is uninformative: cctk_Types.h omits
// the CCTK_REAL2 typedef, while CST still emits CCTK_REAL2 into this thorn's
// generated parameter and argument bindings -- flesh guards *aliased
// function* bindings behind HAVE_CCTK_REAL2, but parameter and grid-variable
// bindings have no such guard (upstream is the same for CCTK_INT16 under
// DISABLE_INT16). The result is a dozen-plus "identifier CCTK_REAL2 is
// undefined" errors, most of them pointing either into a generated
// ParameterCPrivate header or into the fully-expanded (single-line, ~2 kB)
// DECLARE_CCTK_ARGUMENTSX macro, none of them naming the actual cause.
//
// So state the requirement once, here, and include this header first from
// every TestReal2 translation unit.
//
// configure enables HAVE_CCTK_REAL2 when the C compiler provides _Float16,
// which on x86-64 requires GCC >= 12; nvcc-compiled translation units then
// map CCTK_REAL2 to __half instead (see flesh cctk_Types.h). A platform
// whose newest available compiler predates that -- e.g. LSU Deep Bayou,
// gcc 11.2 at most as of 2026-07 -- cannot build this thorn and should drop
// it from its thornlist. That costs no REAL8/REAL4 coverage: the sibling
// thorn TestReal4 carries the whole suite at those two precisions and needs
// no CCTK_REAL2 support to build, as does the rest of the mixed-precision
// stack (CarpetX guards all of its own REAL2 code behind HAVE_CCTK_REAL2).

#include <cctk.h>

#ifndef HAVE_CCTK_REAL2
#error                                                                         \
    "TestReal2 requires a Cactus configuration with CCTK_REAL2 (binary16) support, but HAVE_CCTK_REAL2 is not defined. The flesh's configure enables it when the C compiler provides _Float16, which on x86-64 requires GCC >= 12. Either build with such a compiler, or leave this thorn out of the thornlist (#DISABLED CarpetX/TestReal2) -- its REAL8/REAL4 sibling CarpetX/TestReal4, and the rest of the mixed-precision stack, build correctly without CCTK_REAL2 support."
#endif

#endif // #ifndef CARPETX_TESTREAL2_REQUIRES_REAL2_HXX
