// TestReal4: self-checking mixed-precision test for CarpetX.
//
// Phase 2: two-level regrid + prolongation test. A vertex-centred grid
// function holding a linear function of the coordinates is initialized on
// the (single) coarse level, then a static refined region is created via
// BoxInBox/CarpetXRegrid (see par/testreal4_2lev.par). Because the function
// is linear and CarpetX::prolongation_order = 1, prolongation onto the fine
// level -- performed while filling the fine level's own interior and
// ghosts, as well as the coarse level's ghosts adjacent to the fine level's
// boundary -- reproduces the analytic function exactly, up to
// floating-point roundoff, for CCTK_REAL8, CCTK_REAL4, and CCTK_REAL2 grid
// functions. This directly exercises CarpetX's float/half
// FillPatch_NewLevel/FillPatch_RemakeLevel/FillPatch_ProlongateGhosts and
// the InterpolaterT<CCTK_REAL4>/InterpolaterT<CCTK_REAL2> prolongation
// operators (Phase 2).
//
// Points on the outer physical boundary are excluded from the check (see
// `any(p.NI != 0)` below): the outer boundary uses a dirichlet condition
// with an unrelated constant value (see par/testreal4_2lev.par), which is
// covered by the separate testreal4_bc.par test instead.
//
// This same regrid also drives an edge-centered restriction test (interim
// I2, state8_edgex/y/z etc., see interface.ccl): the state*_edgex/y/z
// groups are initialized with the same linear analytic function, and
// TestReal4_2Lev_EdgeCheck below verifies them on the coarse level after
// the fine level created by the regrid is restricted back down.
//
// It also drives a face-centered restriction test (state8_facex/y/z etc.,
// see interface.ccl), the exact mirror of the edge-centered test above but
// for CarpetX::Restrict's rank-2 (face-centered) path: the state*_facex/y/z
// groups are initialized with the same linear analytic function, and
// TestReal4_2Lev_FaceCheck below verifies them on the coarse level after
// the fine level created by the regrid is restricted back down.

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal4 {

// A function that is linear in each coordinate, hence reproduced exactly
// (up to roundoff) by prolongation of order >= 1.
template <typename T>
constexpr T linear_analytic(const T amplitude, const T x, const T y,
                            const T z) {
  return amplitude * (T(1) + T(2) * x + T(3) * y + T(4) * z);
}

// Amplitudes are chosen merely to make it obvious in debug output which
// grid function a mismatch came from; they carry no other significance.
constexpr CCTK_REAL8 amplitude8_2lev = 1.0;
constexpr CCTK_REAL4 amplitude4_2lev = 2.0f;
// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc): this used to be
// `constexpr CCTK_REAL2 amplitude2_2lev = CCTK_REAL2(1.5);`, referenced
// directly (via `linear_analytic<CCTK_REAL2>`) from inside the
// CCTK_DEVICE lambdas below. Under nvcc, CCTK_REAL2 is `__half` (see
// cctk_Types.h), whose converting constructor from a floating literal is
// not usable in a constant expression as of CUDA 12.2, so that
// declaration would fail to compile under nvcc (it compiles under gcc,
// where CCTK_REAL2 is `_Float16`, whose conversions are constexpr). Keep
// the constant in `float` -- fully constexpr and device-usable on every
// compiler -- and narrow it to CCTK_REAL2 at each call site instead (a
// plain, non-constexpr runtime conversion, exactly like the CCTK_REAL2(p.x)
// conversions already done for the other arguments at those same call
// sites).
constexpr float amplitude2_2lev = 1.5f;

extern "C" void TestReal4_2Lev_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_2Lev_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_2lev(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_2lev(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        // linear_analytic is pure arithmetic (+, *; no library calls), so
        // -- unlike the transcendental-function-based analytic() in
        // testreal4.cxx -- it needs no D5 compute-type indirection:
        // CCTK_REAL2 (_Float16) arithmetic operators are GCC built-ins,
        // not std:: library overloads, so there is no ambiguity to work
        // around.
        u2_2lev(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  // Edge-centered restriction test (interim I2): same amplitudes/analytic
  // function as above (linear_analytic is linear in x, y, *and* z, so it is
  // in particular linear along whichever coordinate direction each of these
  // groups is edge-centered along), just evaluated at the state*_edgex/y/z
  // groups' own centerings (CENTERING={cvv}/{vcv}/{vvc} in interface.ccl).
  // TestReal4_2Lev_EdgeCheck below verifies these on the coarse level after
  // BoxInBox's regrid + restriction.
  grid.loop_int_device<1, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_edgex(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_edgex(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        u2_edgex(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<0, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_edgey(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_edgey(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        u2_edgey(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<0, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_edgez(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_edgez(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        u2_edgez(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  // Face-centered restriction test: same amplitudes/analytic function as
  // above, evaluated at the state*_facex/y/z groups' own centerings
  // (CENTERING={vcc}/{cvc}/{ccv} in interface.ccl). TestReal4_2Lev_FaceCheck
  // below verifies these on the coarse level after BoxInBox's regrid +
  // restriction.
  grid.loop_int_device<0, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_facex(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_facex(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        u2_facex(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_facey(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_facey(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        u2_facey(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_facez(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_facez(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
        u2_facez(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });
}

extern "C" void TestReal4_2Lev_Sync(CCTK_ARGUMENTS) {
  // do nothing -- the SYNC: clauses in schedule.ccl do the actual work
  // (including, on the fine level, filling data by prolongation from the
  // coarse level)
}

extern "C" void TestReal4_2Lev_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_2Lev_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  // Tighter than the single-level periodic test's tolerances would allow
  // (prolongation is exact in exact arithmetic for a linear function), but
  // still loose enough to absorb the roundoff of the prolongation stencil's
  // floating-point arithmetic. CCTK_REAL2's tolerance is looser still and
  // absolute rather than relative: the linear function's magnitude across
  // this parfile's domain/refined region reaches several times the
  // amplitude (amplitude2_2lev * up to ~(1+2+3+4)*0.4 ~ a few), and each
  // term of the prolongation stencil's summation can independently round
  // by up to half's machine epsilon (~9.77e-4) of that magnitude.
  // 4e-2 (not 1e-2): the reference value below is itself evaluated in
  // CCTK_REAL2, and on CUDA (__half device intrinsics) its rounding chain
  // differs from the host's _Float16 one, so host-calibrated 1e-2 (~2.5
  // ulp at the largest checked magnitudes) is missed by up to ~3 ulp on
  // the GPU. 4e-2 is the GPU Validation Plan's ~10-ulp binary16 bound at
  // these amplitudes and still catches any non-roundoff defect.
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;
  constexpr CCTK_REAL8 tolerance2 = 4.0e-2;

  int n_checked = 0;

  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    // Skip points in the outer physical boundary (dirichlet condition with
    // an unrelated constant, checked separately by testreal4_bc.par).
    // `p.NI` is the outward boundary normal and is nonzero only for points
    // in this component's outer-boundary region; coarse-fine (prolongation)
    // ghost zones are always interior to the domain and so have `p.NI == 0`.
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_2lev(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev: state8_2lev::u8_2lev mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_2lev(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev: state4_2lev::u4_2lev mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_2lev(p.I);
    // Widen to double before subtracting/abs: see testreal4.cxx's
    // analytic2 comment for why std::abs(CCTK_REAL2) is itself ambiguous.
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev: state2_2lev::u2_2lev mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked;
  });

  CCTK_VINFO("TestReal4-2lev[state8_2lev]: PASS (%d points checked, level %d)",
             n_checked, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev[state4_2lev]: PASS (%d points checked, level %d)",
             n_checked, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev[state2_2lev]: PASS (%d points checked, level %d)",
             n_checked, cctkGH->cctk_level);
}

// Edge-centered restriction test (interim I2). Unlike TestReal4_2Lev_Check
// above (which verifies prolongation, on every level), this verifies
// CarpetX::Restrict's average-down of edge-centered data, which only
// touches the *coarse* level -- so this function does nothing on the fine
// level (schedule.ccl schedules it, like TestReal4_2Lev_Check, once per
// active level, without OPTIONS: global).
extern "C" void TestReal4_2Lev_EdgeCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_2Lev_EdgeCheck;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  if (cctkGH->cctk_level != 0) {
    CCTK_VINFO(
        "TestReal4-2lev-edge: skipping level %d (only the coarse level, "
        "which restriction actually writes to, is checked)",
        cctkGH->cctk_level);
    return;
  }

  // Same tolerances as TestReal4_2Lev_Check: restriction of a function that
  // is linear along the edge direction is exact in exact arithmetic (it
  // averages two equally-spaced samples of a linear function, reproducing
  // its midpoint value), so only floating-point roundoff of the
  // prolongation (onto the fine level) and average-down (back onto the
  // coarse level) arithmetic remains. tolerance2 is 4e-2 for the same
  // reason as in TestReal4_2Lev_Check: the CCTK_REAL2 reference value's
  // device (__half) rounding chain differs from the host's _Float16 one
  // by a few ulp (first observed as a -0.0117 miss on an A100).
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;
  constexpr CCTK_REAL8 tolerance2 = 4.0e-2;

  int n_checked_x = 0, n_checked_y = 0, n_checked_z = 0;

  grid.loop_all<1, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_edgex(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state8_edgex::u8_edgex mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_edgex(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state4_edgex::u4_edgex mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_edgex(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state2_edgex::u2_edgex mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked_x;
  });

  grid.loop_all<0, 1, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_edgey(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state8_edgey::u8_edgey mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_edgey(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state4_edgey::u4_edgey mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_edgey(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state2_edgey::u2_edgey mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked_y;
  });

  grid.loop_all<0, 0, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_edgez(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state8_edgez::u8_edgez mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_edgez(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state4_edgez::u4_edgez mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_edgez(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev-edge: state2_edgez::u2_edgez mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked_z;
  });

  if (n_checked_x == 0 || n_checked_y == 0 || n_checked_z == 0)
    CCTK_VERROR(
        "TestReal4-2lev-edge: no interior points were found to check on "
        "the coarse level (x: %d, y: %d, z: %d) -- the parfile's regrid "
        "setup is not exercising the edge-centered restriction test",
        n_checked_x, n_checked_y, n_checked_z);

  CCTK_VINFO("TestReal4-2lev-edge[state8_edgex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev-edge[state4_edgex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev-edge[state2_edgex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
}

// Face-centered restriction test. Unlike TestReal4_2Lev_Check (which
// verifies prolongation, on every level), this verifies CarpetX::Restrict's
// average-down of face-centered data, which only touches the *coarse*
// level -- so this function does nothing on the fine level (schedule.ccl
// schedules it, like TestReal4_2Lev_EdgeCheck, once per active level,
// without OPTIONS: global).
extern "C" void TestReal4_2Lev_FaceCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_2Lev_FaceCheck;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  if (cctkGH->cctk_level != 0) {
    CCTK_VINFO(
        "TestReal4-2lev-face: skipping level %d (only the coarse level, "
        "which restriction actually writes to, is checked)",
        cctkGH->cctk_level);
    return;
  }

  // Same tolerances as TestReal4_2Lev_Check/TestReal4_2Lev_EdgeCheck:
  // restriction of a function that is linear along both of a face's
  // transverse (cell-centered) directions is exact in exact arithmetic (it
  // averages equally-spaced samples of a linear function, reproducing its
  // midpoint value), so only floating-point roundoff of the prolongation
  // (onto the fine level) and average-down (back onto the coarse level)
  // arithmetic remains. tolerance2 is 4e-2 for the same reason as in
  // TestReal4_2Lev_Check: the CCTK_REAL2 reference value's device (__half)
  // rounding chain differs from the host's _Float16 one by a few ulp (first
  // observed as a -0.0117 miss on an A100).
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;
  constexpr CCTK_REAL8 tolerance2 = 4.0e-2;

  int n_checked_x = 0, n_checked_y = 0, n_checked_z = 0;

  grid.loop_all<0, 1, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_facex(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev-face: state8_facex::u8_facex mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_facex(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev-face: state4_facex::u4_facex mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_facex(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev-face: state2_facex::u2_facex mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked_x;
  });

  grid.loop_all<1, 0, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_facey(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev-face: state8_facey::u8_facey mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_facey(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev-face: state4_facey::u4_facey mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_facey(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev-face: state2_facey::u2_facey mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked_y;
  });

  grid.loop_all<1, 1, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL8 good8 =
        linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
    const CCTK_REAL8 have8 = u8_facez(p.I);
    const CCTK_REAL8 err8 = have8 - good8;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-2lev-face: state8_facez::u8_facez mismatch at "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected %.17g, error %.17g (tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(good8), double(err8),
          double(tolerance8));

    const CCTK_REAL4 good4 = linear_analytic<CCTK_REAL4>(
        amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y), CCTK_REAL4(p.z));
    const CCTK_REAL4 have4 = u4_facez(p.I);
    const CCTK_REAL4 err4 = have4 - good4;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-2lev-face: state4_facez::u4_facez mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(good4), double(err4),
          double(tolerance4));

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_facez(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal4-2lev-face: state2_facez::u2_facez mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have2), double(good2), double(err2),
          double(tolerance2));

    ++n_checked_z;
  });

  if (n_checked_x == 0 || n_checked_y == 0 || n_checked_z == 0)
    CCTK_VERROR(
        "TestReal4-2lev-face: no interior points were found to check on "
        "the coarse level (x: %d, y: %d, z: %d) -- the parfile's regrid "
        "setup is not exercising the face-centered restriction test",
        n_checked_x, n_checked_y, n_checked_z);

  CCTK_VINFO("TestReal4-2lev-face[state8_facex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev-face[state4_facex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev-face[state2_facex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
}

} // namespace TestReal4
