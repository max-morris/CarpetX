#ifndef CARPETX_CARPETX_INTERP_BASE_HXX
#define CARPETX_CARPETX_INTERP_BASE_HXX

// Precision-generic replacement for amrex::Interpolater.
//
// amrex::Interpolater is hard-wired to amrex::FArrayBox. Note that
// amrex::FArrayBox is a distinct class *derived from*
// amrex::BaseFab<amrex::Real> (== amrex::BaseFab<double>, since AMReX is
// built with amrex::Real == double here; see the static_assert in
// driver.hxx) -- not a type alias for it (see AMReX_FArrayBox.H) -- so an
// amrex::BaseFab<double>& is not itself an amrex::FArrayBox&, even though
// the reverse conversion (FArrayBox& -> BaseFab<double>&, upcasting to the
// base) is free. CarpetX's CCTK_REAL4 grid function groups are stored as
// amrex::fMultiFab (== amrex::FabArray<amrex::BaseFab<float>>, see
// driver.hxx's AnyMultiFab), so cross-level prolongation for those groups
// needs an interpolator whose fab arguments are amrex::BaseFab<float>&, not
// amrex::FArrayBox&. amrex::InterpBase -- the box-only, precision-agnostic
// part of amrex::Interpolater's interface (CoarseBox/BoxCoarsener; see
// AMReX_InterpBase.H) -- has no fab-typed virtuals and is reused unchanged
// as the base for both precisions.
//
// This header is deliberately free-standing (no dependency on driver.hxx):
// driver.hxx needs InterpolaterT<T> for GHExt::...::GroupData::interpolator,
// while prolongate_3d_rf2.hxx (which also needs InterpolaterT<T>) already
// includes driver.hxx. Making InterpolaterT<T> depend on driver.hxx would
// create a cycle.

#include <AMReX_BCRec.H>
#include <AMReX_BaseFab.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuControl.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_IntVect.H>
#include <AMReX_InterpBase.H>
#include <AMReX_Interpolater.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <cctk.h>

#include <cassert>
#include <memory>
#include <type_traits>

namespace amrex {
class Geometry;
}

namespace CarpetX {

// One instantiation of this base exists per storage precision T: T =
// CCTK_REAL for amrex::MultiFab-backed (REAL/REAL8) groups, T = CCTK_REAL4
// for amrex::fMultiFab-backed (REAL4) groups. prolongate_3d_rf2<..., T>
// (prolongate_3d_rf2.hxx) derives from InterpolaterT<T> for both T, instead
// of from amrex::Interpolater.
template <typename T> class InterpolaterT : public amrex::InterpBase {
public:
  InterpolaterT() = default;
  InterpolaterT(const InterpolaterT &) = default;
  InterpolaterT(InterpolaterT &&) = default;
  InterpolaterT &operator=(const InterpolaterT &) = default;
  InterpolaterT &operator=(InterpolaterT &&) = default;
  virtual ~InterpolaterT() override = default;

  // Coarse to fine interpolation in space. Same contract as
  // amrex::Interpolater::interp, except the fab type is amrex::BaseFab<T>
  // (for T=CCTK_REAL, that is amrex::BaseFab<double>, the base class of
  // amrex::FArrayBox, to which amrex::MultiFab's element references
  // implicitly upcast; for T=CCTK_REAL4, amrex::BaseFab<CCTK_REAL4> is
  // amrex::fMultiFab's element type directly). This is a pure virtual
  // function and hence MUST be implemented by derived classes.
  virtual void interp(const amrex::BaseFab<T> &crse, int crse_comp,
                      amrex::BaseFab<T> &fine, int fine_comp, int ncomp,
                      const amrex::Box &fine_region,
                      const amrex::IntVect &ratio,
                      const amrex::Geometry &crse_geom,
                      const amrex::Geometry &fine_geom,
                      amrex::Vector<amrex::BCRec> const &bcr,
                      int actual_comp, int actual_state,
                      amrex::RunOn runon) = 0;

  // Coarse to fine interpolation in space for face-based data.
  //
  // CarpetX's own hand-rolled fill-patch code (fillpatch.cxx) never reaches
  // this, for either precision: it only ever calls the
  // amrex::detail::FillPatchInterp overload that does `mapper->interp(...)`
  // (fillpatch.cxx's 3 FillPatchInterp() call sites), never the
  // InterpFace()/`mapper->interp_face(...)` path that
  // amrex::FillPatchTwoLevels itself uses for face-centred MultiFabs. This
  // is kept here as a non-pure virtual purely for interface parity with
  // amrex::Interpolater::interp_face (which likewise has a non-pure default
  // that Aborts) so that a derived class *may* still override it; if it is
  // ever actually invoked without being overridden, fail loudly rather than
  // silently doing nothing.
  virtual void interp_face(const amrex::BaseFab<T> & /*crse*/,
                           int /*crse_comp*/, amrex::BaseFab<T> & /*fine*/,
                           int /*fine_comp*/, int /*ncomp*/,
                           const amrex::Box & /*fine_region*/,
                           const amrex::IntVect & /*ratio*/,
                           const amrex::IArrayBox & /*solve_mask*/,
                           const amrex::Geometry & /*crse_geom*/,
                           const amrex::Geometry & /*fine_geom*/,
                           amrex::Vector<amrex::BCRec> const & /*bcr*/,
                           int /*bccomp*/, amrex::RunOn /*runon*/) {
    CCTK_ERROR(
        "InterpolaterT<T>::interp_face is not implemented for this "
        "interpolator; face-centred cross-level fill-patch is not "
        "exercised by CarpetX's fillpatch.cxx for either precision");
  }
};

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_INTERP_BASE_HXX
