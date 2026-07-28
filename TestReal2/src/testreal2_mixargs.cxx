// TestReal2_MixArgs: PROVIDES FUNCTION coverage for the aliased-function
// sized-REAL grammar/binding-generation gap (see interface.ccl's Phase 5
// section, and repos/flesh/src/piraha/pegs/interface.peg's
// ret_type/arg_type rules) in the case that involves CCTK_REAL2. TestReal4
// carries the REAL2-free half of this coverage (TestReal4_MixArgs, a
// CCTK_REAL4 return with CCTK_REAL8/CCTK_REAL4 arguments), so the machinery
// stays covered on platforms without CCTK_REAL2 support.
//
// This translation unit implements the provider (WITH
// TestReal2_MixArgs_Impl in interface.ccl's PROVIDES FUNCTION clause) for
// the CCTK_REAL4 FUNCTION TestReal2_MixArgs alias, which takes a
// CCTK_REAL8, a CCTK_REAL4, and a CCTK_REAL2 argument. It combines all
// three with distinct weights (1, 2, 4) into a single CCTK_REAL4 result,
// so that a mistake in argument order, width, or value -- anywhere along
// the aliased-function machinery (grammar parsing, CreateFunctionBindings
// generation, or the actual call) -- would change the returned value.
//
// src/testreal2_mixargs_check.cxx (a different translation unit) both
// USES and calls this alias, and verifies the returned value.

#include "testreal2_requires_real2.hxx"

#include <cctk.h>

namespace TestReal2 {

extern "C" CCTK_REAL4 TestReal2_MixArgs_Impl(const CCTK_REAL8 a,
                                             const CCTK_REAL4 b,
                                             const CCTK_REAL2 c) {
  // Widen each argument to CCTK_REAL4 (a narrowing for a, a no-op for b, a
  // widening for c -- binary16 -> float32 is always exact, no precision
  // loss) before combining, so the arithmetic itself is unambiguous.
  return CCTK_REAL4(a) + CCTK_REAL4(2) * b + CCTK_REAL4(4) * CCTK_REAL4(c);
}

} // namespace TestReal2
