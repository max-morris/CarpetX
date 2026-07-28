// TestReal2: two-level regrid + prolongation test, and the edge-/face-
// centered restriction tests, at CCTK_REAL2 (binary16). The
// CCTK_REAL8/CCTK_REAL4 counterparts live in the sibling thorn TestReal4
// (src/testreal4_2lev.cxx), which documents the setup in full; in brief:
//
// A vertex-centred grid function holding a function that is linear in the
// coordinates is initialized on the (single) coarse level, then a static
// refined region is created via BoxInBox/CarpetXRegrid (see
// par/testreal2_2lev.par). Because the function is linear and
// CarpetX::prolongation_order = 1, prolongation onto the fine level
// reproduces the analytic function exactly, up to floating-point roundoff.
// This exercises CarpetX's half FillPatch_NewLevel/FillPatch_RemakeLevel/
// FillPatch_ProlongateGhosts and the InterpolaterT<CCTK_REAL2> prolongation
// operator. Points on the outer physical boundary are excluded from the
// check (`any(p.NI != 0)` below), since that boundary carries an unrelated
// dirichlet constant covered by testreal2_bc.par instead.
//
// The same regrid also drives an edge-centered restriction test
// (state2_edgex/y/z, interim I2: CarpetX's local FAB-templated edge
// average-down) and a face-centered one (state2_facex/y/z, which for REAL2
// goes through CarpetX's local __half-clean average_down_faces_local rather
// than stock amrex::average_down_faces -- see interface.ccl).

#include "testreal2_requires_real2.hxx"

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal2 {

// A function that is linear in each coordinate, hence reproduced exactly
// (up to roundoff) by prolongation of order >= 1. Kept in sync with the
// identical definition in TestReal4's testreal4_2lev.cxx.
template <typename T>
constexpr T linear_analytic(const T amplitude, const T x, const T y,
                            const T z) {
  return amplitude * (T(1) + T(2) * x + T(3) * y + T(4) * z);
}

// The amplitude is chosen merely to make it obvious in debug output which
// grid function a mismatch came from; it carries no other significance.
//
// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc): this used to be
// `constexpr CCTK_REAL2 amplitude2_2lev = CCTK_REAL2(1.5);`, referenced
// directly (via `linear_analytic<CCTK_REAL2>`) from inside the CCTK_DEVICE
// lambdas below. Under nvcc, CCTK_REAL2 is `__half` (see cctk_Types.h),
// whose converting constructor from a floating literal is not usable in a
// constant expression as of CUDA 12.2, so that declaration would fail to
// compile under nvcc (it compiles under gcc, where CCTK_REAL2 is
// `_Float16`, whose conversions are constexpr). Keep the constant in
// `float` -- fully constexpr and device-usable on every compiler -- and
// narrow it to CCTK_REAL2 at each call site instead (a plain,
// non-constexpr runtime conversion, exactly like the CCTK_REAL2(p.x)
// conversions already done for the other arguments at those same call
// sites).
constexpr float amplitude2_2lev = 1.5f;

extern "C" void TestReal2_2Lev_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_2Lev_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        // linear_analytic is pure arithmetic (+, *; no library calls), so
        // -- unlike the transcendental-function-based analytic() in
        // testreal2.cxx -- it needs no D5 compute-type indirection:
        // CCTK_REAL2 (_Float16) arithmetic operators are GCC built-ins,
        // not std:: library overloads, so there is no ambiguity to work
        // around.
        u2_2lev(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  // Edge-centered restriction test (interim I2): same amplitude/analytic
  // function as above (linear_analytic is linear in x, y, *and* z, so it is
  // in particular linear along whichever coordinate direction each of these
  // groups is edge-centered along), just evaluated at the state2_edgex/y/z
  // groups' own centerings (CENTERING={cvv}/{vcv}/{vvc} in interface.ccl).
  // TestReal2_2Lev_EdgeCheck below verifies these on the coarse level after
  // BoxInBox's regrid + restriction.
  grid.loop_int_device<1, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_edgex(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<0, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_edgey(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<0, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_edgez(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  // Face-centered restriction test: same amplitude/analytic function as
  // above, evaluated at the state2_facex/y/z groups' own centerings
  // (CENTERING={vcc}/{cvc}/{ccv} in interface.ccl). TestReal2_2Lev_FaceCheck
  // below verifies these on the coarse level after regrid + restriction.
  grid.loop_int_device<0, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_facex(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_facey(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });

  grid.loop_int_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_facez(p.I) = linear_analytic<CCTK_REAL2>(
            CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
            CCTK_REAL2(p.z));
      });
}

extern "C" void TestReal2_2Lev_Sync(CCTK_ARGUMENTS) {
  // do nothing -- the SYNC: clauses in schedule.ccl do the actual work
  // (including, on the fine level, filling data by prolongation from the
  // coarse level)
}

// The CCTK_REAL2 tolerance used by all three checks below. Absolute rather
// than relative, and much looser than TestReal4's REAL8/REAL4 tolerances:
// the linear function's magnitude across this parfile's domain/refined
// region reaches several times the amplitude (amplitude2_2lev *
// up to ~(1+2+3+4)*0.4 ~ a few), and each term of the prolongation
// stencil's summation can independently round by up to half's machine
// epsilon (~9.77e-4) of that magnitude.
// 4e-2 (not 1e-2): the reference value in each check is itself evaluated in
// CCTK_REAL2, and on CUDA (__half device intrinsics) its rounding chain
// differs from the host's _Float16 one, so host-calibrated 1e-2 (~2.5 ulp
// at the largest checked magnitudes) is missed by up to ~3 ulp on the GPU
// (first observed as a -0.0117 miss on an A100). 4e-2 is the GPU Validation
// Plan's ~10-ulp binary16 bound at these amplitudes and still catches any
// non-roundoff defect.
constexpr CCTK_REAL8 tolerance2_2lev = 4.0e-2;

extern "C" void TestReal2_2Lev_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_2Lev_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  int n_checked = 0;

  grid.loop_all<0, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    // Skip points in the outer physical boundary (dirichlet condition with
    // an unrelated constant, checked separately by testreal2_bc.par).
    // `p.NI` is the outward boundary normal and is nonzero only for points
    // in this component's outer-boundary region; coarse-fine (prolongation)
    // ghost zones are always interior to the domain and so have `p.NI == 0`.
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_2lev(p.I);
    // Widen to double before subtracting/abs: see testreal2.cxx's
    // analytic2 comment for why std::abs(CCTK_REAL2) is itself ambiguous.
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev: state2_2lev::u2_2lev mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked;
  });

  CCTK_VINFO("TestReal2-2lev[state2_2lev]: PASS (%d points checked, level %d)",
             n_checked, cctkGH->cctk_level);
}

// Edge-centered restriction test (interim I2). Unlike TestReal2_2Lev_Check
// above (which verifies prolongation, on every level), this verifies
// CarpetX::Restrict's average-down of edge-centered data, which only
// touches the *coarse* level -- so this function does nothing on the fine
// level (schedule.ccl schedules it, like TestReal2_2Lev_Check, once per
// active level, without OPTIONS: global).
extern "C" void TestReal2_2Lev_EdgeCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_2Lev_EdgeCheck;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  if (cctkGH->cctk_level != 0) {
    CCTK_VINFO(
        "TestReal2-2lev-edge: skipping level %d (only the coarse level, "
        "which restriction actually writes to, is checked)",
        cctkGH->cctk_level);
    return;
  }

  // Restriction of a function that is linear along the edge direction is
  // exact in exact arithmetic (it averages two equally-spaced samples of a
  // linear function, reproducing its midpoint value), so only floating-point
  // roundoff of the prolongation (onto the fine level) and average-down
  // (back onto the coarse level) arithmetic remains.
  int n_checked_x = 0, n_checked_y = 0, n_checked_z = 0;

  grid.loop_all<1, 0, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_edgex(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev-edge: state2_edgex::u2_edgex mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked_x;
  });

  grid.loop_all<0, 1, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_edgey(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev-edge: state2_edgey::u2_edgey mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked_y;
  });

  grid.loop_all<0, 0, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_edgez(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev-edge: state2_edgez::u2_edgez mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked_z;
  });

  if (n_checked_x == 0 || n_checked_y == 0 || n_checked_z == 0)
    CCTK_VERROR(
        "TestReal2-2lev-edge: no interior points were found to check on "
        "the coarse level (x: %d, y: %d, z: %d) -- the parfile's regrid "
        "setup is not exercising the edge-centered restriction test",
        n_checked_x, n_checked_y, n_checked_z);

  CCTK_VINFO("TestReal2-2lev-edge[state2_edgex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
}

// Face-centered restriction test. Unlike TestReal2_2Lev_Check (which
// verifies prolongation, on every level), this verifies CarpetX::Restrict's
// average-down of face-centered data, which only touches the *coarse*
// level -- so this function does nothing on the fine level (schedule.ccl
// schedules it, like TestReal2_2Lev_EdgeCheck, once per active level,
// without OPTIONS: global).
extern "C" void TestReal2_2Lev_FaceCheck(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_2Lev_FaceCheck;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  if (cctkGH->cctk_level != 0) {
    CCTK_VINFO(
        "TestReal2-2lev-face: skipping level %d (only the coarse level, "
        "which restriction actually writes to, is checked)",
        cctkGH->cctk_level);
    return;
  }

  // Restriction of a function that is linear along both of a face's
  // transverse (cell-centered) directions is exact in exact arithmetic (it
  // averages equally-spaced samples of a linear function, reproducing its
  // midpoint value), so only floating-point roundoff of the prolongation
  // (onto the fine level) and average-down (back onto the coarse level)
  // arithmetic remains.
  int n_checked_x = 0, n_checked_y = 0, n_checked_z = 0;

  grid.loop_all<0, 1, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_facex(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev-face: state2_facex::u2_facex mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked_x;
  });

  grid.loop_all<1, 0, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_facey(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev-face: state2_facey::u2_facey mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked_y;
  });

  grid.loop_all<1, 1, 0>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    if (any(p.NI != 0))
      return;

    const CCTK_REAL2 good2 = linear_analytic<CCTK_REAL2>(
        CCTK_REAL2(amplitude2_2lev), CCTK_REAL2(p.x), CCTK_REAL2(p.y),
        CCTK_REAL2(p.z));
    const CCTK_REAL2 have2 = u2_facez(p.I);
    const CCTK_REAL8 err2 = double(have2) - double(good2);
    if (abs(err2) > tolerance2_2lev)
      CCTK_VERROR(
          "TestReal2-2lev-face: state2_facez::u2_facez mismatch at "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(good2), double(err2),
          double(tolerance2_2lev));

    ++n_checked_z;
  });

  if (n_checked_x == 0 || n_checked_y == 0 || n_checked_z == 0)
    CCTK_VERROR(
        "TestReal2-2lev-face: no interior points were found to check on "
        "the coarse level (x: %d, y: %d, z: %d) -- the parfile's regrid "
        "setup is not exercising the face-centered restriction test",
        n_checked_x, n_checked_y, n_checked_z);

  CCTK_VINFO("TestReal2-2lev-face[state2_facex/y/z]: PASS (%d/%d/%d points "
             "checked, level %d)",
             n_checked_x, n_checked_y, n_checked_z, cctkGH->cctk_level);
}

} // namespace TestReal2
