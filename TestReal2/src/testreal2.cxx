// TestReal2: the CCTK_REAL2 (binary16) layer of the mixed-precision test
// suite whose CCTK_REAL8/CCTK_REAL4 layer is the sibling thorn TestReal4.
//
// Single-level periodic ghost-sync self-test: state2 is filled with a
// deterministic analytic function that is mathematically exactly periodic
// with period 1 in each direction, synced, and then verified everywhere
// (interior and ghost zones). The parameter file sets up a domain whose
// extent in each direction is an integer multiple of 1 (see
// par/testreal2.par), so the analytic value at a ghost point (whose
// coordinate lies outside [xmin,xmax]) and the value at its periodic image
// inside the domain agree up to the rounding error of the trigonometric
// functions -- which only holds if single-level ghost sync is correct for a
// CCTK_REAL2 grid function. TestReal4's testreal4.cxx does the same for
// CCTK_REAL8/CCTK_REAL4 (and, in testreal4_gf3d5.cxx, re-checks REAL4
// through the GF3D5 accessor family).
//
// This translation unit is compiled with the default accessor family, i.e.
// CARPETX_GF3D5 is not defined, so the CCTK_CENTERING_GF macro (expanded
// inside DECLARE_CCTK_ARGUMENTSX_<function>) produces GF3D2 accessors.

#include "testreal2_requires_real2.hxx"

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>
#include <type_traits>

namespace TestReal2 {

// Kept in sync with (and deliberately named differently from) the identical
// definition in TestReal4's testreal4.cxx, so the two thorns' translation
// units cannot violate the One Definition Rule.
template <typename T>
constexpr T analytic(const T amplitude, const T x, const T y, const T z) {
  using std::acos, std::cos;
  const T two_pi = 2 * acos(-T(1));
  return amplitude * cos(two_pi * x) * cos(two_pi * y) * cos(two_pi * z);
}

// The amplitude is chosen merely to make it obvious in debug output which
// grid function a mismatch came from; it carries no other significance.
//
// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc): this used to be
// `constexpr CCTK_REAL2 amplitude2 = CCTK_REAL2(1.5);`, referenced directly
// from inside the CCTK_DEVICE lambda in TestReal2_Initialize below. Under
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
// issue documented for std::isnan in CarpetX's valid.cxx check_valid_gf).
// So the REAL2 analytic value is computed in `float` (which does have an
// unambiguous std::cos/acos overload), and only the final result is
// narrowed to CCTK_REAL2.
// C3 (mixed precision, CCTK_REAL2): nvcc rejects calling a plain __host__
// function from inside a CCTK_DEVICE lambda ("calling a __host__ function
// ... is not allowed"). `analytic2` is called from the CCTK_DEVICE lambda in
// TestReal2_Initialize below (and, harmlessly, host-side too), so it needs
// both annotations.
static CCTK_DEVICE CCTK_HOST CCTK_REAL2 analytic2(const float amplitude,
                                                  const float x, const float y,
                                                  const float z) {
  return CCTK_REAL2(analytic<float>(amplitude, x, y, z));
}

extern "C" void TestReal2_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2(p.I) = analytic2(amplitude2, float(p.x), float(p.y), float(p.z));
      });
}

extern "C" void TestReal2_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_Check;
  DECLARE_CCTK_PARAMETERS;

  // Compile-time proof that DECLARE_CCTK_PARAMETERS binds a sized-REAL
  // param.ccl parameter at exactly its declared width, not merely to a
  // value that happens to convert correctly at runtime. (TestReal4 makes
  // the same assertion for check_tolerance8/check_tolerance4.)
  static_assert(std::is_same_v<std::remove_const_t<std::remove_reference_t<
                                   decltype(check_tolerance2)>>,
                               CCTK_REAL2>);

  using std::abs;

  // The tolerance is taken from param.ccl (check_tolerance2) rather than
  // hardcoded, so this test also exercises the sized-REAL parameter
  // machinery end-to-end: the analytic function is evaluated with different
  // floating-point operations (and hence different rounding) at Initialize
  // time (device loop) than at Check time (host loop), and at a ghost
  // point's own (extrapolated) coordinate rather than at its periodic
  // image's coordinate, so exact bit-for-bit equality is not expected --
  // only agreement to within the precision of the type. binary16 has only
  // 10 mantissa bits (machine epsilon ~9.77e-4), so even a single rounding
  // step can move the amplitude-~1.5-scaled result by a few times 1e-3.
  const CCTK_REAL8 tolerance2 = double(check_tolerance2);

  // par/testreal2.par explicitly steers check_tolerance2 away from its
  // param.ccl default (0.00390625 = 2^-8) to 0.0078125 (2^-7); both values
  // are exact powers of two, hence exactly representable in binary16, so
  // the value read back here from the parameter database (after having
  // gone through param.ccl's range check and CCTK_REAL2 storage) must
  // match the parfile's override bit-for-bit -- proving the
  // parfile-parsing path round-trips a CCTK_REAL2 parameter correctly, not
  // just approximately. (Every parfile that enables run_periodic_test --
  // testreal2.par, testreal2_io.par, testreal2_io_recover.par,
  // testreal2_plotfile.par -- must set check_tolerance2 = 0.0078125
  // explicitly to match this check.)
  if (CCTK_REAL2(check_tolerance2) != CCTK_REAL2(0.0078125))
    CCTK_VERROR(
        "TestReal2: check_tolerance2 = %.9g, expected the parfile-steered "
        "value 0.0078125 (2^-7) -- the CCTK_REAL2 parameter did not "
        "round-trip through par/testreal2.par correctly",
        double(check_tolerance2));

  int n_checked = 0;

  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
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
          "TestReal2: state2::u2 mismatch at (%.9g,%.9g,%.9g) "
          "(level %d, patch %d, component %d): have %.9g, expected %.9g, "
          "error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2), double(tolerance2));

    ++n_checked;
  });

  CCTK_VINFO("TestReal2[state2]: PASS (%d points)", n_checked);
}

// CarpetX's checkpoint/recovery only restores each group's interior (not
// its ghost zones or outer boundary), so every group needs re-syncing
// once after recovery, before any poststep/analysis routine (e.g.
// TestReal2_Check, which reads "everywhere") runs again. This function is
// a no-op: the SYNC clause on its schedule.ccl entry (AT
// post_recover_variables) does the actual work, exactly like
// TestOutput_Sync in TestOutput/src/TestOutput.cxx.
extern "C" void TestReal2_PostRecover_Sync(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_PostRecover_Sync;
}

} // namespace TestReal2
