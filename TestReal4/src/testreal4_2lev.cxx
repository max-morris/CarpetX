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
// floating-point roundoff, for CCTK_REAL8 and CCTK_REAL4 grid functions.
// This directly exercises CarpetX's float
// FillPatch_NewLevel/FillPatch_RemakeLevel/FillPatch_ProlongateGhosts and
// the InterpolaterT<CCTK_REAL4> prolongation operator (Phase 2). TestReal2,
// the sibling thorn holding this suite's CCTK_REAL2 layer, does the same at
// binary16.
//
// Points on the outer physical boundary are excluded from the check (see
// `any(p.NI != 0)` below): the outer boundary uses a dirichlet condition
// with an unrelated constant value (see par/testreal4_2lev.par), which is
// covered by the separate testreal4_bc.par test instead.
//
// This same regrid also drives an edge-centered restriction test (interim
// I2, state8_edgex/y/z and state4_edgex/y/z, see interface.ccl): the state*_edgex/y/z
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
      });

  grid.loop_int_device<0, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_edgey(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_edgey(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
      });

  grid.loop_int_device<0, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_edgez(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_edgez(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
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
      });

  grid.loop_int_device<1, 0, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_facey(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_facey(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
      });

  grid.loop_int_device<1, 1, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_facez(p.I) =
            linear_analytic<CCTK_REAL8>(amplitude8_2lev, p.x, p.y, p.z);
        u4_facez(p.I) = linear_analytic<CCTK_REAL4>(
            amplitude4_2lev, CCTK_REAL4(p.x), CCTK_REAL4(p.y),
            CCTK_REAL4(p.z));
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
  // floating-point arithmetic.
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;

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

    ++n_checked;
  });

  CCTK_VINFO("TestReal4-2lev[state8_2lev]: PASS (%d points checked, level %d)",
             n_checked, cctkGH->cctk_level);
  CCTK_VINFO("TestReal4-2lev[state4_2lev]: PASS (%d points checked, level %d)",
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
  // coarse level) arithmetic remains.
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;

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
  // arithmetic remains.
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;

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
}

} // namespace TestReal4
