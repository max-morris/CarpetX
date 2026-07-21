// TestReal4: grid scalar / grid array sized-real coverage.
//
// Phase 4: exercises CarpetX's AnyTypeVector-backed grid scalar
// (TYPE=scalar) and grid array (TYPE=array) storage (driver.hxx) at all
// three sized-REAL precisions (REAL8/REAL4/REAL2) -- allocation, poisoning,
// validity checking, checkpoint/recovery, and I/O output (TSV/openPMD/Silo)
// -- complementing this thorn's grid function (TYPE=gf) coverage in
// testreal4.cxx/testreal4_2lev.cxx/testreal4_bc.cxx/testreal4_evol.cxx.
//
// Every value written below is an exact binary fraction (an integer, or an
// integer plus a multiple of 1/2 or 1/4), with magnitude well under
// binary16's ~2048 exact-integer range, so it is representable bit-for-bit
// at every precision: TestReal4_ScalarArray_Check compares bit-exactly, with
// no rounding tolerance needed (unlike the trigonometric analytic function
// used for the grid-function periodic test in testreal4.cxx).

#include <cctk.h>
#include <cctk_Arguments.h>

namespace TestReal4 {

constexpr int scalar_array_size = 16;

extern "C" void TestReal4_ScalarArray_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_ScalarArray_Initialize;

  *scalar8 = CCTK_REAL8(3.75);
  *scalar4 = CCTK_REAL4(6.25);
  *scalar2 = CCTK_REAL2(-2.5);

  for (int i = 0; i < scalar_array_size; ++i) {
    array8[i] = CCTK_REAL8(1) + CCTK_REAL8(i) * CCTK_REAL8(0.25);
    array4[i] = CCTK_REAL4(2) + CCTK_REAL4(i) * CCTK_REAL4(0.5);
    // Pure CCTK_REAL2 arithmetic (_Float16 addition/multiplication): a GCC
    // built-in operator, not a std:: library call, so (unlike e.g. cos() in
    // testreal4.cxx's analytic2) there is no overload-resolution ambiguity
    // to work around here.
    array2[i] = CCTK_REAL2(-1) + CCTK_REAL2(i) * CCTK_REAL2(0.5);
  }
}

extern "C" void TestReal4_ScalarArray_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_ScalarArray_Check;

  if (*scalar8 != CCTK_REAL8(3.75))
    CCTK_VERROR("TestReal4: grid scalar REAL8 \"scalar8\" mismatch: have "
                "%.17g, expected %.17g",
                double(*scalar8), 3.75);
  if (*scalar4 != CCTK_REAL4(6.25))
    CCTK_VERROR("TestReal4: grid scalar REAL4 \"scalar4\" mismatch: have "
                "%.9g, expected %.9g",
                double(*scalar4), 6.25);
  if (*scalar2 != CCTK_REAL2(-2.5))
    CCTK_VERROR("TestReal4: grid scalar REAL2 \"scalar2\" mismatch: have "
                "%.9g, expected %.9g",
                double(*scalar2), -2.5);

  for (int i = 0; i < scalar_array_size; ++i) {
    const CCTK_REAL8 good8 = CCTK_REAL8(1) + CCTK_REAL8(i) * CCTK_REAL8(0.25);
    if (array8[i] != good8)
      CCTK_VERROR("TestReal4: grid array REAL8 \"array8\"[%d] mismatch: have "
                  "%.17g, expected %.17g",
                  i, double(array8[i]), double(good8));

    const CCTK_REAL4 good4 = CCTK_REAL4(2) + CCTK_REAL4(i) * CCTK_REAL4(0.5);
    if (array4[i] != good4)
      CCTK_VERROR("TestReal4: grid array REAL4 \"array4\"[%d] mismatch: have "
                  "%.9g, expected %.9g",
                  i, double(array4[i]), double(good4));

    const CCTK_REAL2 good2 = CCTK_REAL2(-1) + CCTK_REAL2(i) * CCTK_REAL2(0.5);
    if (array2[i] != good2)
      CCTK_VERROR("TestReal4: grid array REAL2 \"array2\"[%d] mismatch: have "
                  "%.9g, expected %.9g",
                  i, double(array2[i]), double(good2));
  }

  CCTK_VINFO("TestReal4[scalar REAL8 scalar8]: PASS");
  CCTK_VINFO("TestReal4[scalar REAL4 scalar4]: PASS");
  CCTK_VINFO("TestReal4[scalar REAL2 scalar2]: PASS");
  CCTK_VINFO("TestReal4[array REAL8 array8]: PASS (%d elements)",
             scalar_array_size);
  CCTK_VINFO("TestReal4[array REAL4 array4]: PASS (%d elements)",
             scalar_array_size);
  CCTK_VINFO("TestReal4[array REAL2 array2]: PASS (%d elements)",
             scalar_array_size);
}

} // namespace TestReal4
