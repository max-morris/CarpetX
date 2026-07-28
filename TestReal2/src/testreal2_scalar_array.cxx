// TestReal2: grid scalar / grid array coverage at CCTK_REAL2 (binary16); the
// CCTK_REAL8/CCTK_REAL4 counterpart is in the sibling thorn TestReal4
// (src/testreal4_scalar_array.cxx).
//
// Phase 4: exercises CarpetX's AnyTypeVector-backed grid scalar
// (TYPE=scalar) and grid array (TYPE=array) storage (driver.hxx) at binary16
// -- allocation, poisoning, validity checking, checkpoint/recovery, and I/O
// output (TSV/openPMD/Silo) -- complementing this thorn's grid function
// (TYPE=gf) coverage in testreal2.cxx/testreal2_2lev.cxx/testreal2_bc.cxx/
// testreal2_evol.cxx.
//
// Every value written below is an exact binary fraction (an integer, or an
// integer plus a multiple of 1/2), with magnitude well under binary16's
// ~2048 exact-integer range, so it is representable bit-for-bit:
// TestReal2_ScalarArray_Check compares bit-exactly, with no rounding
// tolerance needed (unlike the trigonometric analytic function used for the
// grid-function periodic test in testreal2.cxx).

#include "testreal2_requires_real2.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>

namespace TestReal2 {

constexpr int scalar_array_size = 16;

extern "C" void TestReal2_ScalarArray_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_ScalarArray_Initialize;

  *scalar2 = CCTK_REAL2(-2.5);

  for (int i = 0; i < scalar_array_size; ++i) {
    // Pure CCTK_REAL2 arithmetic (_Float16 addition/multiplication): a GCC
    // built-in operator, not a std:: library call, so (unlike e.g. cos() in
    // testreal2.cxx's analytic2) there is no overload-resolution ambiguity
    // to work around here.
    array2[i] = CCTK_REAL2(-1) + CCTK_REAL2(i) * CCTK_REAL2(0.5);
  }
}

extern "C" void TestReal2_ScalarArray_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_ScalarArray_Check;

  if (*scalar2 != CCTK_REAL2(-2.5))
    CCTK_VERROR("TestReal2: grid scalar REAL2 \"scalar2\" mismatch: have "
                "%.9g, expected %.9g",
                double(*scalar2), -2.5);

  for (int i = 0; i < scalar_array_size; ++i) {
    const CCTK_REAL2 good2 = CCTK_REAL2(-1) + CCTK_REAL2(i) * CCTK_REAL2(0.5);
    if (array2[i] != good2)
      CCTK_VERROR("TestReal2: grid array REAL2 \"array2\"[%d] mismatch: have "
                  "%.9g, expected %.9g",
                  i, double(array2[i]), double(good2));
  }

  CCTK_VINFO("TestReal2[scalar REAL2 scalar2]: PASS");
  CCTK_VINFO("TestReal2[array REAL2 array2]: PASS (%d elements)",
             scalar_array_size);
}

} // namespace TestReal2
