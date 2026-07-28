// TestReal4_MixArgs_Check: USES-side counterpart of
// src/testreal4_mixargs.cxx's TestReal4_MixArgs_Impl provider. This
// translation unit calls the CCTK_REAL4 FUNCTION TestReal4_MixArgs alias
// (interface.ccl's Phase 5 section) with known CCTK_REAL8/CCTK_REAL4
// argument values, and verifies the returned CCTK_REAL4 against the same
// weighted-sum computation performed here independently in double precision
// -- exercising the aliased-function machinery's handling of sized-REAL
// return types and mixed sized-REAL argument types together in a single
// call, round-tripped through an actual call at runtime. TestReal2's
// testreal2_mixargs_check.cxx does the same for the CCTK_REAL2-taking
// variant of the alias.
//
// Scheduled (schedule.ccl, gated behind run_periodic_test so it runs in
// the same parfiles as the original periodic self-test, notably
// par/testreal4.par) as a global routine that touches no grid variables,
// so it needs no DECLARE_CCTK_PARAMETERS and no grid loop.

#include <cctk.h>
#include <cctk_Arguments.h>

#include <cmath>

namespace TestReal4 {

extern "C" void TestReal4_MixArgs_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_MixArgs_Check;

  // Known argument values, chosen to be exactly representable at their
  // declared width, so the "expected" value computed below is not itself
  // subject to any representation error, and only genuine round-trip
  // mistakes (wrong argument order/width/value, or a binding-generation
  // bug) can make `have` disagree with it.
  const CCTK_REAL8 a = 3.0;
  const CCTK_REAL4 b = 1.25f;

  // The actual aliased-function call: TestReal4_MixArgs is the USES
  // FUNCTION alias name (interface.ccl), not the WITH provider name
  // (TestReal4_MixArgs_Impl) -- its prototype is generated into
  // TestReal4_Prototypes.h and pulled in transitively via cctk.h.
  const CCTK_REAL4 have = TestReal4_MixArgs(a, b);

  // Mirror TestReal4_MixArgs_Impl's weighted sum (weights 1, 2 on the
  // REAL8 and REAL4 arguments respectively), independently, in double
  // precision.
  const double expected = double(a) + 2.0 * double(b);

  // Both contributions are exact in float at the chosen argument values, so
  // the tolerance only guards against a future change to a or b without
  // needing to be tightened.
  const double tolerance = 4.0 * 1.19e-7;

  const double have_d = double(have);
  const double err = have_d - expected;
  using std::abs;
  if (abs(err) > tolerance)
    CCTK_VERROR(
        "TestReal4-aliasfn: TestReal4_MixArgs(a=%.17g, b=%.9g) = "
        "%.9g, expected %.17g (error %.17g, tolerance %.17g)",
        double(a), double(b), have_d, expected, err, tolerance);

  CCTK_VINFO("TestReal4-aliasfn: PASS");
}

} // namespace TestReal4
