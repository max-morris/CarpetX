// TestReal4: self-checking mixed-precision test for CarpetX.
//
// Phase 2: physical (dirichlet) boundary condition test. A cell-centred
// grid function is initialized to a value that differs from its dirichlet
// value, then synced (which applies the physical boundary conditions, see
// GroupData::apply_boundary_conditions and BoundaryCondition<T> in
// boundaries.cxx/boundaries_impl.hxx). The check then verifies that every
// point on the outer physical boundary carries exactly the configured
// dirichlet constant, for both a CCTK_REAL8 and a CCTK_REAL4 grid function,
// exercising the templated BoundaryCondition<CCTK_REAL4> code path.
// (TestReal2, the sibling thorn holding this suite's CCTK_REAL2 layer, does
// the same for BoundaryCondition<CCTK_REAL2>.)
//
// The dirichlet constants below must match the `dirichlet_values` TAGS on
// state8_bc/state4_bc/state2_bc in interface.ccl.

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal4 {

// Must match the TAGS='dirichlet_values={...}' entries in interface.ccl.
constexpr CCTK_REAL8 dirichlet_value8_bc = 7.0;
constexpr CCTK_REAL4 dirichlet_value4_bc = 4.25f;

// Initial interior value, deliberately different from the dirichlet values
// above so that a boundary condition failure (e.g. the dirichlet condition
// silently not being applied, leaving the initial value or poison in place)
// is unambiguously detected.
constexpr CCTK_REAL8 initial_value8_bc = 0.0;
constexpr CCTK_REAL4 initial_value4_bc = 0.0f;

extern "C" void TestReal4_BC_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_BC_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u8_bc(p.I) = initial_value8_bc;
        u4_bc(p.I) = initial_value4_bc;
      });
}

extern "C" void TestReal4_BC_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal4_BC_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  // The dirichlet condition stores the constant directly (converted to the
  // grid function's element type), so agreement should be exact up to the
  // roundoff of that single conversion.
  constexpr CCTK_REAL8 tolerance8 = 1.0e-12;
  constexpr CCTK_REAL4 tolerance4 = 1.0e-6f;

  int n_boundary_checked = 0;

  grid.loop_all<1, 1, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    // Only outer-boundary points are dirichlet points; `p.NI` (the outward
    // boundary normal) is nonzero exactly there.
    if (!any(p.NI != 0))
      return;

    const CCTK_REAL8 have8 = u8_bc(p.I);
    const CCTK_REAL8 err8 = have8 - dirichlet_value8_bc;
    if (abs(err8) > tolerance8)
      CCTK_VERROR(
          "TestReal4-bc: state8_bc::u8_bc mismatch at outer boundary point "
          "(%.17g,%.17g,%.17g) (level %d, patch %d, component %d): have "
          "%.17g, expected dirichlet value %.17g, error %.17g "
          "(tolerance %.17g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have8), double(dirichlet_value8_bc),
          double(err8), double(tolerance8));

    const CCTK_REAL4 have4 = u4_bc(p.I);
    const CCTK_REAL4 err4 = have4 - dirichlet_value4_bc;
    if (abs(err4) > tolerance4)
      CCTK_VERROR(
          "TestReal4-bc: state4_bc::u4_bc mismatch at outer boundary point "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected dirichlet value %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch,
          p.component, double(have4), double(dirichlet_value4_bc),
          double(err4), double(tolerance4));

    ++n_boundary_checked;
  });

  if (n_boundary_checked == 0)
    CCTK_VERROR("TestReal4-bc: no outer-boundary points were found to check "
                "-- the parfile's boundary setup is not exercising this "
                "test");

  CCTK_VINFO("TestReal4-bc[state8_bc]: PASS (%d boundary points checked)",
             n_boundary_checked);
  CCTK_VINFO("TestReal4-bc[state4_bc]: PASS (%d boundary points checked)",
             n_boundary_checked);
}

} // namespace TestReal4
