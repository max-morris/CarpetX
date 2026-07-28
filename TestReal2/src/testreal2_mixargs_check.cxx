// TestReal2_MixArgs_Check: USES-side counterpart of
// src/testreal2_mixargs.cxx's TestReal2_MixArgs_Impl provider. This
// translation unit calls the CCTK_REAL4 FUNCTION TestReal2_MixArgs alias
// (interface.ccl's Phase 5 section) with known CCTK_REAL8/CCTK_REAL4/
// CCTK_REAL2 argument values, and verifies the returned CCTK_REAL4 against
// the same weighted-sum computation performed here independently in
// double precision -- exercising the aliased-function machinery's
// handling of sized-REAL return types and mixed sized-REAL argument types
// together in a single call, round-tripped through an actual call at
// runtime.
//
// Scheduled (schedule.ccl, gated behind the shared TestReal4::
// run_periodic_test so it runs in the same parfiles as the periodic
// self-test, notably par/testreal2.par) as a global routine that touches no
// grid variables, so it needs no DECLARE_CCTK_PARAMETERS and no grid loop.
//
// C/C++ only, per interface.ccl's comment: CCTK_REAL2 has no Fortran
// spelling, so TestReal2_MixArgs must never be called from Fortran; there
// is no Fortran caller anywhere in this coverage.

#include "testreal2_requires_real2.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>

#include <cmath>

namespace TestReal2 {

extern "C" void TestReal2_MixArgs_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_MixArgs_Check;

  // Known argument values, chosen to be exactly representable at their
  // declared width -- including CCTK_REAL2's binary16 (0.5 = 2^-1) -- so
  // the "expected" value computed below is not itself subject to any
  // representation error, and only genuine round-trip mistakes (wrong
  // argument order/width/value, or a binding-generation bug) can make
  // `have` disagree with it.
  const CCTK_REAL8 a = 3.0;
  const CCTK_REAL4 b = 1.25f;
  const CCTK_REAL2 c = CCTK_REAL2(0.5);

  // The actual aliased-function call: TestReal2_MixArgs is the USES
  // FUNCTION alias name (interface.ccl), not the WITH provider name
  // (TestReal2_MixArgs_Impl) -- its prototype is generated into
  // TestReal2_Prototypes.h and pulled in transitively via cctk.h.
  const CCTK_REAL4 have = TestReal2_MixArgs(a, b, c);

  // Mirror TestReal2_MixArgs_Impl's weighted sum (weights 1, 2, 4 on the
  // REAL8, REAL4, and REAL2 arguments respectively), independently, in
  // double precision.
  const double expected = double(a) + 2.0 * double(b) + 4.0 * double(c);

  // The REAL8/REAL4 contributions (a, b) are exact in float; only the
  // REAL2 contribution (weight 4 on c) is given any tolerance, sized to a
  // few times CCTK_REAL2's machine epsilon (binary16 has 10 mantissa
  // bits, epsilon ~9.77e-4) -- even though the chosen value of c is
  // itself an exact power of two and so introduces no actual rounding
  // here; the tolerance guards against a future change to c's value
  // without needing to be tightened.
  const double tolerance = 4.0 * 4.0 * 9.77e-4;

  const double have_d = double(have);
  const double err = have_d - expected;
  using std::abs;
  if (abs(err) > tolerance)
    CCTK_VERROR(
        "TestReal2-aliasfn: TestReal2_MixArgs(a=%.17g, b=%.9g, c=%.6g) = "
        "%.9g, expected %.17g (error %.17g, tolerance %.17g)",
        double(a), double(b), double(c), have_d, expected, err, tolerance);

  CCTK_VINFO("TestReal2-aliasfn: PASS");
}

} // namespace TestReal2
