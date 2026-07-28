// TestReal2: physical (dirichlet) boundary condition test at CCTK_REAL2
// (binary16); the CCTK_REAL8/CCTK_REAL4 counterpart is in the sibling thorn
// TestReal4 (src/testreal4_bc.cxx).
//
// A cell-centred grid function is initialized to a value that differs from
// its dirichlet value, then synced (which applies the physical boundary
// conditions, see CarpetX's GroupData::apply_boundary_conditions and
// BoundaryCondition<T> in boundaries.cxx/boundaries_impl.hxx). The check
// then verifies that every point on the outer physical boundary carries
// exactly the configured dirichlet constant, exercising the templated
// BoundaryCondition<CCTK_REAL2> code path.
//
// The dirichlet constant below must match the `dirichlet_values` TAGS on
// state2_bc in interface.ccl.

#include "testreal2_requires_real2.hxx"

#include <loop_device.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>

namespace TestReal2 {

// Must match the TAGS='dirichlet_values={...}' entry in interface.ccl.
// 3.5 is exactly representable in binary16 (needs only 2 mantissa bits),
// so the single conversion applied when storing the dirichlet constant
// introduces no rounding at all -- the exact-representable value D9 asks
// for where feasible.
// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc): this used to be
// `constexpr CCTK_REAL2 dirichlet_value2_bc = CCTK_REAL2(3.5);`. Under
// nvcc, CCTK_REAL2 is `__half` (see cctk_Types.h), whose converting
// constructor from a floating literal is not usable in a constant
// expression as of CUDA 12.2, so that declaration would fail to compile
// under nvcc (it compiles under gcc, where CCTK_REAL2 is `_Float16`,
// whose conversions are constexpr). Unlike initial_value2_bc below, this
// constant is read only from TestReal2_BC_Check, a host-only function
// (grid.loop_all, no CCTK_DEVICE lambda) -- so it needs no compile-time
// (device-embeddable) value, and simply demoting it to a runtime-
// initialized `const` (identical value, computed once at static
// initialization instead of at compile time) is sufficient and CPU-safe.
const CCTK_REAL2 dirichlet_value2_bc = CCTK_REAL2(3.5);

// Initial interior value, deliberately different from the dirichlet value
// above so that a boundary condition failure (e.g. the dirichlet condition
// silently not being applied, leaving the initial value or poison in place)
// is unambiguously detected.
// H2: unlike dirichlet_value2_bc above, this constant *is* read from
// inside the CCTK_DEVICE lambda in TestReal2_BC_Initialize below, so
// demoting it to plain `const` would not work (a non-constexpr host
// global is not accessible from device code). Keep it in `float` instead
// -- fully constexpr and device-usable on every compiler, and exactly
// representable in binary16 as well (0.0) -- and narrow it to CCTK_REAL2
// at the point of use, a plain runtime conversion.
constexpr float initial_value2_bc = 0.0f;

extern "C" void TestReal2_BC_Initialize(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_BC_Initialize;
  DECLARE_CCTK_PARAMETERS;

  grid.loop_int_device<1, 1, 1>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        u2_bc(p.I) = CCTK_REAL2(initial_value2_bc);
      });
}

extern "C" void TestReal2_BC_Check(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_TestReal2_BC_Check;
  DECLARE_CCTK_PARAMETERS;

  using std::abs;

  // The dirichlet condition stores the constant directly (converted to the
  // grid function's element type), so agreement should be exact up to the
  // roundoff of that single conversion. dirichlet_value2_bc = 3.5 is
  // exactly representable in binary16, so that conversion is also exact in
  // principle; the tolerance is still kept far looser than TestReal4's
  // REAL8/REAL4 ones as a safety margin against any intermediate widening
  // this driver's CCTK_REAL2 boundary-condition code path may perform (D5).
  constexpr CCTK_REAL8 tolerance2 = 1.0e-3;

  int n_boundary_checked = 0;

  grid.loop_all<1, 1, 1>(grid.nghostzones, [&](const Loop::PointDesc &p) {
    // Only outer-boundary points are dirichlet points; `p.NI` (the outward
    // boundary normal) is nonzero exactly there.
    if (!any(p.NI != 0))
      return;

    const CCTK_REAL2 have2 = u2_bc(p.I);
    // Widen to double before subtracting/abs -- see testreal2.cxx's
    // analytic2 comment for why std::abs(CCTK_REAL2) is itself ambiguous.
    const CCTK_REAL8 err2 = double(have2) - double(dirichlet_value2_bc);
    if (abs(err2) > tolerance2)
      CCTK_VERROR(
          "TestReal2-bc: state2_bc::u2_bc mismatch at outer boundary point "
          "(%.9g,%.9g,%.9g) (level %d, patch %d, component %d): have %.9g, "
          "expected dirichlet value %.9g, error %.9g (tolerance %.9g)",
          double(p.x), double(p.y), double(p.z), p.level, p.patch, p.component,
          double(have2), double(dirichlet_value2_bc), double(err2),
          double(tolerance2));

    ++n_boundary_checked;
  });

  if (n_boundary_checked == 0)
    CCTK_VERROR("TestReal2-bc: no outer-boundary points were found to check "
                "-- the parfile's boundary setup is not exercising this "
                "test");

  CCTK_VINFO("TestReal2-bc[state2_bc]: PASS (%d boundary points checked)",
             n_boundary_checked);
}

} // namespace TestReal2
