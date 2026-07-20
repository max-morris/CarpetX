// TestReal4: self-checking mixed-precision test for CarpetX.
//
// This translation unit is compiled with the *default* accessor family,
// i.e. CARPETX_GF3D5 is *not* defined, so the CCTK_CENTERING_GF macro
// (expanded inside DECLARE_CCTK_ARGUMENTSX_<function>) produces GF3D2
// accessors -- see src/testreal4_gf3d5.cxx for the GF3D5 counterpart.

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal4 {

// A smooth function of the coordinates that is mathematically exactly
// periodic with period 1 in each direction. The parameter file sets up a
// domain whose extent in each direction is an integer multiple of 1 (see
// par/testreal4.par), so the analytic value at a ghost point (whose
// coordinate lies outside [xmin,xmax]) and the value at its periodic image
// inside the domain agree up to the rounding error of the trigonometric
// functions. TestReal4_Check verifies this for every point, including
// ghost zones, which only works if single-level ghost sync is correct for
// both a CCTK_REAL8 and a CCTK_REAL4 grid function.
template <typename T>
constexpr T analytic(const T amplitude, const T x, const T y, const T z) {
  using std::acos, std::cos;
  const T two_pi = 2 * acos(-T(1));
  return amplitude * cos(two_pi * x) * cos(two_pi * y) * cos(two_pi * z);
}

// Amplitudes are chosen merely to make it obvious in debug output which
// grid function a mismatch came from; they carry no other significance.
constexpr CCTK_REAL8 amplitude8 = 1.0;
constexpr CCTK_REAL4 amplitude4 = 2.0f;

extern "C" void TestReal4_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8(p.I) = analytic<CCTK_REAL8>(amplitude8, p.x, p.y, p.z);
        u4(p.I) = analytic<CCTK_REAL4>(amplitude4, CCTK_REAL4(p.x),
                                        CCTK_REAL4(p.y), CCTK_REAL4(p.z));
      });
}

extern "C" void TestReal4_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  // Type-appropriate tolerances: the analytic function is evaluated with
  // different floating-point operations (and hence different rounding) at
  // Initialize time (device loop) than at Check time (host loop), and at a
  // ghost point's own (extrapolated) coordinate rather than at its
  // periodic image's coordinate, so exact bit-for-bit equality is not
  // expected -- only agreement to within the precision of the type.
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-6f;

  int n_checked = 0;

  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    const CCTK_REAL8 good8 = analytic<CCTK_REAL8>(amplitude8, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4: state8::u8 mismatch at (%.17g,%.17g,%.17g) "
          "(level %d, patch %d, component %d): have %.17g, expected %.17g, "
          "error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = analytic<CCTK_REAL4>(
        amplitude4, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4: state4::u4 mismatch at (%.9g,%.9g,%.9g) "
          "(level %d, patch %d, component %d): have %.9g, expected %.9g, "
          "error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    ++n_checked;
  });

  CCTK_VINFO("TestReal4: PASS (%d points)", n_checked);
}

} // namespace TestReal4
