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
constexpr CCTK_REAL2 amplitude2_2lev = CCTK_REAL2(1.5);

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
            amplitude2_2lev, CCTK_REAL2(p.x), CCTK_REAL2(p.y),
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
  constexpr CCTK_REAL8 tolerance8 = 1.0e-9;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;
  constexpr CCTK_REAL8 tolerance2 = 1.0e-2;

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
        amplitude2_2lev, CCTK_REAL2(p.x), CCTK_REAL2(p.y), CCTK_REAL2(p.z));
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

} // namespace TestReal4
