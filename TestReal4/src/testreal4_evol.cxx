// TestReal4: self-checking mixed-precision test for CarpetX.
//
// Phase 2: ODESolvers evolution smoke test. Two independent, trivially
// exactly-solvable exponential-decay ODEs
//
//   du/dt = -decay_rate * u,     u(0) = u0,     u(t) = u0 exp(-decay_rate t)
//
// are evolved by ODESolvers (see par/testreal4_evol.par, which selects RK4),
// one state held as a CCTK_REAL8 grid function (state8_evol/rhs8_evol), one
// as a CCTK_REAL4 grid function (state4_evol/rhs4_evol), and one as a
// CCTK_REAL2 grid function (state2_evol/rhs2_evol). This exercises
// ODESolvers::statecomp_t's per-group mixed-precision support (each (state,
// rhs) pair keeps its own element type throughout the lincomb/
// combine_valids kernels).
//
// The RK4 time truncation error and the CCTK_REAL4/CCTK_REAL2 rounding
// error are the dominant error sources here (not exactly zero, unlike the
// polynomial prolongation test), so the checks below use tolerances
// appropriate for a converged, but not bitwise-exact, numerical
// integration: double ~1e-12 (par/testreal4_evol.par uses a small enough
// CarpetX::dtfac that the O(dt^4) RK4 truncation error stays well below
// this), float ~1e-5 relative (dominated by CCTK_REAL4 rounding, not by
// truncation error), and half ~5e-3 relative (dominated by CCTK_REAL2
// rounding accumulated over dtfac^-1 RK4 steps, each of which rounds its
// state to binary16 -- notably looser than the periodic/2lev/bc tests'
// REAL2 tolerances because this is the one REAL2 test where rounding
// error actually *accumulates* step over step rather than being applied
// once).

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal4 {

extern "C" void TestReal4_Evol_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Evol_Initial;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL8 u0_8 = evol_u0;
  const CCTK_REAL4 u0_4 = CCTK_REAL4(evol_u0);
  const CCTK_REAL2 u0_2 = CCTK_REAL2(evol_u0);

  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_evol(p.I) = u0_8;
        u4_evol(p.I) = u0_4;
        u2_evol(p.I) = u0_2;
        // rhs8_evol/rhs4_evol/rhs2_evol are only ever *computed* (in
        // TestReal4_Evol_RHS) when this parfile's run_evol_test enables
        // ODESolvers to drive the evolution. But CarpetX grants storage
        // for every declared group unconditionally (schedule.ccl STORAGE
        // is not actually gate-able in this driver -- see the note at the
        // top of schedule.ccl), so in the other three parfiles rhs8_evol/
        // rhs4_evol/rhs2_evol would otherwise be left unwritten and fail
        // CarpetX's validity check at the very first CycleTimelevels.
        // Zero-init them here too; their value is never asserted on
        // outside the evol test.
        r8_evol(p.I) = CCTK_REAL8(0);
        r4_evol(p.I) = CCTK_REAL4(0);
        r2_evol(p.I) = CCTK_REAL2(0);
      });
}

extern "C" void TestReal4_Evol_PostStep(CCTK_ARGUMENTS) {
  // do nothing -- the SYNC: clause in schedule.ccl does the actual work
}

extern "C" void TestReal4_PostRecover_ReinitRHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_PostRecover_ReinitRHS;

  // rhs8_evol/rhs4_evol/rhs2_evol are TAGS='checkpoint="no"', so
  // checkpoint/recovery never writes (or re-validates) them: CarpetX's
  // recovery path marks every group invalid before reading the
  // checkpoint, and only groups actually present in the checkpoint get
  // their interior re-validated by that read, so these three groups are
  // left invalid ("Recovering") even though TestReal4_Evol_Initial had
  // already zero-filled (and validated) them earlier in this same
  // recovery's CCTK_INITIAL re-run. Zero-fill them again here, for exactly
  // the same reason (and with exactly the same value) as
  // TestReal4_Evol_Initial above, so that TestReal4_PostRecover_Sync's
  // SYNC of these groups (schedule.ccl, AT post_recover_variables) has a
  // validly-written interior to work from.
  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        r8_evol(p.I) = CCTK_REAL8(0);
        r4_evol(p.I) = CCTK_REAL4(0);
        r2_evol(p.I) = CCTK_REAL2(0);
      });
}

extern "C" void TestReal4_Evol_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Evol_RHS;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL8 lambda8 = evol_decay_rate;
  const CCTK_REAL4 lambda4 = CCTK_REAL4(evol_decay_rate);
  const CCTK_REAL2 lambda2 = CCTK_REAL2(evol_decay_rate);

  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        r8_evol(p.I) = -lambda8 * u8_evol(p.I);
        r4_evol(p.I) = -lambda4 * u4_evol(p.I);
        // Pure CCTK_REAL2 arithmetic (negation, multiplication): a GCC
        // built-in operator, not a std:: library call, so (unlike exp()
        // below) there is no overload-resolution ambiguity to work around
        // here -- see testreal4.cxx's analytic2 comment.
        r2_evol(p.I) = -lambda2 * u2_evol(p.I);
      });
}

extern "C" void TestReal4_Evol_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_Evol_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs, std::exp;

  const CCTK_REAL8 exact8 =
      CCTK_REAL8(evol_u0) * exp(-CCTK_REAL8(evol_decay_rate) * cctk_time);
  const CCTK_REAL4 exact4 = CCTK_REAL4(evol_u0) *
                            exp(-CCTK_REAL4(evol_decay_rate) *
                                CCTK_REAL4(cctk_time));
  // D5 compute-type indirection: exp() has no unambiguous CCTK_REAL2
  // overload (see testreal4.cxx's analytic2 comment), and there is no
  // reason to want one here anyway -- the analytic solution's reference
  // value is computed in double, exactly like exact8 above, and only
  // compared against `have2` after widening (below), never narrowed to
  // CCTK_REAL2 itself.
  const CCTK_REAL8 exact2 =
      CCTK_REAL8(evol_u0) * exp(-CCTK_REAL8(evol_decay_rate) * cctk_time);

  constexpr CCTK_REAL8 tolerance8 = 1.0e-12;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-5f;
  constexpr CCTK_REAL8 tolerance2 = 5.0e-3;

  int n_checked = 0;

  grid.loop_all<1, 1, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    const CCTK_REAL8 have8 = u8_evol(p.I);
    const CCTK_REAL8 scale8 = abs(exact8) > 0 ? abs(exact8) : CCTK_REAL8(1);
    const CCTK_REAL8 relerr8 = abs(have8 - exact8) / scale8;
    if (relerr8 > tolerance8)
      CCTK_VERROR(
          "TestReal4-evol: state8_evol::u8_evol mismatch at t=%.17g: have "
          "%.17g, expected %.17g, relative error %.17g (tolerance %.17g)",
          double(cctk_time), double(have8), double(exact8), double(relerr8),
          double(tolerance8));

    const CCTK_REAL4 have4 = u4_evol(p.I);
    const CCTK_REAL4 scale4 = abs(exact4) > 0 ? abs(exact4) : CCTK_REAL4(1);
    const CCTK_REAL4 relerr4 = abs(have4 - exact4) / scale4;
    if (relerr4 > tolerance4)
      CCTK_VERROR(
          "TestReal4-evol: state4_evol::u4_evol mismatch at t=%.9g: have "
          "%.9g, expected %.9g, relative error %.9g (tolerance %.9g)",
          double(cctk_time), double(have4), double(exact4), double(relerr4),
          double(tolerance4));

    // have2 is widened to double before any arithmetic/abs, exactly like
    // exact2 above -- never a bare CCTK_REAL2 std:: call.
    const CCTK_REAL8 have2 = double(u2_evol(p.I));
    const CCTK_REAL8 scale2 = abs(exact2) > 0 ? abs(exact2) : CCTK_REAL8(1);
    const CCTK_REAL8 relerr2 = abs(have2 - exact2) / scale2;
    if (relerr2 > tolerance2)
      CCTK_VERROR(
          "TestReal4-evol: state2_evol::u2_evol mismatch at t=%.9g: have "
          "%.9g, expected %.9g, relative error %.9g (tolerance %.9g)",
          double(cctk_time), double(have2), double(exact2), double(relerr2),
          double(tolerance2));

    ++n_checked;
  });

  CCTK_VINFO("TestReal4-evol[state8_evol]: PASS (%d points checked, t=%.6g, "
             "state8=%.15g)",
             n_checked, double(cctk_time), double(exact8));
  CCTK_VINFO("TestReal4-evol[state4_evol]: PASS (%d points checked, t=%.6g, "
             "state4=%.7g)",
             n_checked, double(cctk_time), double(exact4));
  CCTK_VINFO("TestReal4-evol[state2_evol]: PASS (%d points checked, t=%.6g, "
             "state2=%.7g)",
             n_checked, double(cctk_time), double(exact2));
}

} // namespace TestReal4
