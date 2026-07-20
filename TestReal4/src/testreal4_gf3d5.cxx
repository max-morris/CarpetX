// TestReal4: self-checking mixed-precision test for CarpetX.
//
// This translation unit defines CARPETX_GF3D5 *before* including the loop
// headers, so the CCTK_CENTERING_GF macro (expanded inside
// DECLARE_CCTK_ARGUMENTSX_<function>) produces GF3D5 accessors instead of
// the default GF3D2 ones -- see src/testreal4.cxx for the GF3D2 branch.
// This exercises the CARPETX_GF3D5 code path in Loop/src/loop.hxx with a
// CCTK_REAL4 grid function.
#define CARPETX_GF3D5

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal4 {

// Keep this in sync with the (otherwise identical) definition in
// testreal4.cxx. It is deliberately given a different name so that the two
// translation units cannot accidentally violate the One Definition Rule.
template <typename T>
constexpr T analytic_gf3d5(const T amplitude, const T x, const T y,
                           const T z) {
  using std::acos, std::cos;
  const T two_pi = 2 * acos(-T(1));
  return amplitude * cos(two_pi * x) * cos(two_pi * y) * cos(two_pi * z);
}

// Must match the amplitude4 used in testreal4.cxx to initialize state4::u4.
constexpr CCTK_REAL4 amplitude4 = 2.0f;

extern "C" void TestReal4_CheckGF3D5(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_CheckGF3D5;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  constexpr CCTK_REAL4 tolerance4 = 1.0e-6f;

  int n_checked = 0;

  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    const CCTK_REAL4 good4 = analytic_gf3d5<CCTK_REAL4>(
        amplitude4, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4(cctk_layout_VVV, p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4 (GF3D5 accessor): state4::u4 mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    ++n_checked;
  });

  CCTK_VINFO("TestReal4 (GF3D5 accessor): PASS (%d points)", n_checked);
}

} // namespace TestReal4
