// TestReal2: ODESolvers evolution smoke test at CCTK_REAL2 (binary16); the
// CCTK_REAL8/CCTK_REAL4 counterpart is in the sibling thorn TestReal4
// (src/testreal4_evol.cxx), and both are steered by the same shared
// TestReal4::evol_decay_rate / TestReal4::evol_u0 parameters, so running
// the two thorns together evolves the same ODE at all three precisions.
//
// The trivially exactly-solvable exponential-decay ODE
//
//   du/dt = -decay_rate * u,     u(0) = u0,     u(t) = u0 exp(-decay_rate t)
//
// is evolved by ODESolvers (see par/testreal2_evol.par, which selects RK4)
// with the state held as a CCTK_REAL2 grid function (state2_evol/rhs2_evol),
// exercising ODESolvers::statecomp_t's per-group mixed-precision support
// (each (state, rhs) pair keeps its own element type throughout the
// lincomb/combine_valids kernels).
//
// The REAL2 tolerance below (~5e-3 relative) is dominated by binary16
// rounding accumulated over dtfac^-1 RK4 steps, each of which rounds its
// state to binary16 -- notably looser than the periodic/2lev/bc tests'
// REAL2 tolerances because this is the one REAL2 test where rounding error
// actually *accumulates* step over step rather than being applied once.

#include "testreal2_requires_real2.hxx"

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal2 {

extern "C" void TestReal2_Evol_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_Evol_Initial;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL2 u0_2 = CCTK_REAL2(evol_u0);

  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_evol(p.I) = u0_2;
        // rhs2_evol is only ever *computed* (in TestReal2_Evol_RHS) when
        // this parfile's run_evol_test enables ODESolvers to drive the
        // evolution. But CarpetX grants storage for every declared group
        // unconditionally (schedule.ccl STORAGE is not actually gate-able
        // in this driver -- see the note at the top of schedule.ccl), so in
        // the other parfiles rhs2_evol would otherwise be left unwritten
        // and fail CarpetX's validity check at the very first
        // CycleTimelevels. Zero-init it here too; its value is never
        // asserted on outside the evol test.
        r2_evol(p.I) = CCTK_REAL2(0);
      });
}

extern "C" void TestReal2_Evol_PostStep(CCTK_ARGUMENTS) {
  // do nothing -- the SYNC: clause in schedule.ccl does the actual work
}

extern "C" void TestReal2_PostRecover_ReinitRHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_PostRecover_ReinitRHS;

  // rhs2_evol is TAGS='checkpoint="no"', so checkpoint/recovery never
  // writes (or re-validates) it: CarpetX's recovery path marks every group
  // invalid before reading the checkpoint, and only groups actually present
  // in the checkpoint get their interior re-validated by that read, so this
  // group is left invalid ("Recovering") even though TestReal2_Evol_Initial
  // had already zero-filled (and validated) it earlier in this same
  // recovery's CCTK_INITIAL re-run. Zero-fill it again here, for exactly
  // the same reason (and with exactly the same value) as
  // TestReal2_Evol_Initial above, so that TestReal2_PostRecover_Sync's
  // SYNC of this group (schedule.ccl, AT post_recover_variables) has a
  // validly-written interior to work from.
  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        r2_evol(p.I) = CCTK_REAL2(0);
      });
}

extern "C" void TestReal2_Evol_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_Evol_RHS;
  DECLARE_CCTK_PARAMETERS;

  const CCTK_REAL2 lambda2 = CCTK_REAL2(evol_decay_rate);

  grid.loop_all_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        // Pure CCTK_REAL2 arithmetic (negation, multiplication): a GCC
        // built-in operator, not a std:: library call, so (unlike exp()
        // below) there is no overload-resolution ambiguity to work around
        // here -- see testreal2.cxx's analytic2 comment.
        r2_evol(p.I) = -lambda2 * u2_evol(p.I);
      });
}

extern "C" void TestReal2_Evol_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_Evol_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs, std::exp;

  // D5 compute-type indirection: exp() has no unambiguous CCTK_REAL2
  // overload (see testreal2.cxx's analytic2 comment), and there is no
  // reason to want one here anyway -- the analytic solution's reference
  // value is computed in double, exactly like TestReal4's exact8, and only
  // compared against `have2` after widening (below), never narrowed to
  // CCTK_REAL2 itself.
  const CCTK_REAL8 exact2 =
      CCTK_REAL8(evol_u0) * exp(-CCTK_REAL8(evol_decay_rate) * cctk_time);

  constexpr CCTK_REAL8 tolerance2 = 5.0e-3;

  int n_checked = 0;

  grid.loop_all<1, 1, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    // have2 is widened to double before any arithmetic/abs, exactly like
    // exact2 above -- never a bare CCTK_REAL2 std:: call.
    const CCTK_REAL8 have2 = double(u2_evol(p.I));
    const CCTK_REAL8 scale2 = abs(exact2) > 0 ? abs(exact2) : CCTK_REAL8(1);
    const CCTK_REAL8 relerr2 = abs(have2 - exact2) / scale2;
    if (relerr2 > tolerance2)
      CCTK_VERROR(
          "TestReal2-evol: state2_evol::u2_evol mismatch at t=%.9g: have "
          "%.9g, expected %.9g, relative error %.9g (tolerance %.9g)",
          double(cctk_time), double(have2), double(exact2), double(relerr2),
          double(tolerance2));

    ++n_checked;
  });

  CCTK_VINFO("TestReal2-evol[state2_evol]: PASS (%d points checked, t=%.6g, "
             "state2=%.7g)",
             n_checked, double(cctk_time), double(exact2));
}

} // namespace TestReal2
