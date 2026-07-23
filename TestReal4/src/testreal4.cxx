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
#include <type_traits>

namespace TestReal4 {

// A smooth function of the coordinates that is mathematically exactly
// periodic with period 1 in each direction. The parameter file sets up a
// domain whose extent in each direction is an integer multiple of 1 (see
// par/testreal4.par), so the analytic value at a ghost point (whose
// coordinate lies outside [xmin,xmax]) and the value at its periodic image
// inside the domain agree up to the rounding error of the trigonometric
// functions. TestReal4_Check verifies this for every point, including
// ghost zones, which only works if single-level ghost sync is correct for
// a CCTK_REAL8, a CCTK_REAL4, and a CCTK_REAL2 grid function.
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
// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc): this used to be
// `constexpr CCTK_REAL2 amplitude2 = CCTK_REAL2(1.5);`, referenced directly
// from inside the CCTK_DEVICE lambda in TestReal4_Initialize below. Under
// nvcc, CCTK_REAL2 is `__half` (see cctk_Types.h), whose converting
// constructor from a floating literal is not usable in a constant
// expression as of CUDA 12.2, so that declaration would fail to compile
// under nvcc (it compiles under gcc, where CCTK_REAL2 is `_Float16`, whose
// conversions are constexpr). `analytic2` immediately widens its amplitude
// to `float` anyway (see its comment below), so we simply keep the
// constant in `float` -- fully constexpr and device-usable on every
// compiler -- and let `analytic2` do the (runtime, on-device) narrowing
// conversion to CCTK_REAL2 that it already does internally for its result.
constexpr float amplitude2 = 1.5f;

// D5 compute-type indirection: `analytic<T>` above calls std::acos/cos,
// which (unlike ordinary arithmetic on CCTK_REAL2 = `_Float16`, which GCC
// built-in-promotes to `float` under the hood per D1) have no unambiguous
// overload for a bare `_Float16` argument -- overload resolution among the
// library's float/double/long double overloads is ambiguous (the same
// issue documented for std::isnan in valid.cxx's check_valid_gf). So the
// REAL2 analytic value is computed in `float` (which does have an
// unambiguous std::cos/acos overload), and only the final result is
// narrowed to CCTK_REAL2.
static CCTK_REAL2 analytic2(const float amplitude, const float x,
                            const float y, const float z) {
  return CCTK_REAL2(analytic<float>(amplitude, x, y, z));
}

extern "C" void TestReal4_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8(p.I) = analytic<CCTK_REAL8>(amplitude8, p.x, p.y, p.z);
        u4(p.I) = analytic<CCTK_REAL4>(amplitude4, CCTK_REAL4(p.x),
                                        CCTK_REAL4(p.y), CCTK_REAL4(p.z));
        u2(p.I) = analytic2(amplitude2, float(p.x), float(p.y), float(p.z));
      });
}

extern "C" void TestReal4_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Check;
  DECLARE_CCTK_PARAMETERS;

  // Compile-time proof that DECLARE_CCTK_PARAMETERS binds each sized-REAL
  // param.ccl parameter at exactly its declared width, not merely to a
  // value that happens to convert correctly at runtime.
  static_assert(std::is_same_v<std::remove_const_t<std::remove_reference_t<
                                    decltype(check_tolerance8)>>,
                                CCTK_REAL8>);
  static_assert(std::is_same_v<std::remove_const_t<std::remove_reference_t<
                                    decltype(check_tolerance4)>>,
                                CCTK_REAL4>);
  static_assert(std::is_same_v<std::remove_const_t<std::remove_reference_t<
                                    decltype(check_tolerance2)>>,
                                CCTK_REAL2>);

  using std::abs;

  // Type-appropriate tolerances, taken from param.ccl (check_tolerance8/4/2)
  // rather than hardcoded, so this test also exercises the sized-REAL
  // parameter machinery end-to-end (parsed, range-checked, stored, and
  // bound at its declared width): the analytic function is evaluated with
  // different floating-point operations (and hence different rounding) at
  // Initialize time (device loop) than at Check time (host loop), and at a
  // ghost point's own (extrapolated) coordinate rather than at its
  // periodic image's coordinate, so exact bit-for-bit equality is not
  // expected -- only agreement to within the precision of the type.
  // CCTK_REAL2's tolerance is loosest by far: binary16 has only 10
  // mantissa bits (machine epsilon ~9.77e-4), so even a single rounding
  // step can move the amplitude-~1.5-scaled result by a few times 1e-3.
  const CCTK_REAL8 tolerance8 = check_tolerance8;
  const CCTK_REAL4 tolerance4 = check_tolerance4;
  const CCTK_REAL8 tolerance2 = double(check_tolerance2);

  // par/testreal4.par explicitly steers check_tolerance2 away from its
  // param.ccl default (0.00390625 = 2^-8) to 0.0078125 (2^-7); both values
  // are exact powers of two, hence exactly representable in binary16, so
  // the value read back here from the parameter database (after having
  // gone through param.ccl's range check and CCTK_REAL2 storage) must
  // match the parfile's override bit-for-bit -- proving the
  // parfile-parsing path round-trips a CCTK_REAL2 parameter correctly, not
  // just approximately. (Every parfile that enables run_periodic_test --
  // testreal4.par, testreal4_io.par, testreal4_io_recover.par -- must set
  // check_tolerance2 = 0.0078125 explicitly to match this check.)
  if (CCTK_REAL2(check_tolerance2) != CCTK_REAL2(0.0078125))
    CCTK_VERROR(
        "TestReal4: check_tolerance2 = %.9g, expected the parfile-steered "
        "value 0.0078125 (2^-7) -- the CCTK_REAL2 parameter did not "
        "round-trip through par/testreal4.par correctly",
        double(check_tolerance2));

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

    // D5: compare in double (via analytic2's float computation, narrowed
    // to CCTK_REAL2), rather than calling std::abs on a bare CCTK_REAL2 --
    // see the analytic2 comment above for why std::abs(CCTK_REAL2) would
    // itself be ambiguous.
    const CCTK_REAL2 good2 =
        analytic2(amplitude2, float(p.x), float(p.y), float(p.z));
    const CCTK_REAL2 have2 = u2(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4: state2::u2 mismatch at (%.9g,%.9g,%.9g) "
          "(level %d, patch %d, component %d): have %.9g, expected %.9g, "
          "error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked;
  });

  CCTK_VINFO("TestReal4[state8]: PASS (%d points)", n_checked);
  CCTK_VINFO("TestReal4[state4]: PASS (%d points)", n_checked);
  CCTK_VINFO("TestReal4[state2]: PASS (%d points)", n_checked);
}

// CarpetX's checkpoint/recovery only restores each group's interior (not
// its ghost zones or outer boundary), so every group needs re-syncing
// once after recovery, before any poststep/analysis routine (e.g.
// TestReal4_Check, which reads "everywhere") runs again. This function is
// a no-op: the SYNC clause on its schedule.ccl entry (AT
// post_recover_variables) does the actual work, exactly like
// TestOutput_Sync in TestOutput/src/TestOutput.cxx.
extern "C" void TestReal4_PostRecover_Sync(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_PostRecover_Sync;
}

} // namespace TestReal4
