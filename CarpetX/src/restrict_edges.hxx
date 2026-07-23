#ifndef CARPETX_CARPETX_RESTRICT_EDGES_HXX
#define CARPETX_CARPETX_RESTRICT_EDGES_HXX

// Interim: local FAB-templated edge average-down, plus a local FAB-templated
// face average-down usable for CCTK_REAL2 (__half). Upstream goal has two
// parts:
//  (1) template amrex::average_down_edges over FAB (mirroring
//      average_down_faces); drop average_down_edges/average_down_edges_kernel
//      below when the ET's AMReX includes that.
//  (2) fix amrex::amrex_avgdown_faces's point kernel (used by AMReX's own
//      FAB-templated average_down_faces<FAB>, AMReX_MultiFabUtil_3D_C.H) to
//      be __half-clean: it computes `T facInv = T(1.0) / (facy*facz);`,
//      i.e. `T / int`, which is ambiguous for T = __half under nvcc (only
//      MIXED __half/int operands are ambiguous -- see the file-level compile
//      notes referenced from schedule.cxx). This is a trivial upstream AMReX
//      PR candidate: `T(1.0) / (facy*facz)` -> `T(1.0) / T(facy*facz)` (and
//      likewise for the facx*facz and facx*facy cases). Until that lands, we
//      cannot use amrex::average_down_faces<BaseFab<CCTK_REAL2>> directly, so
//      average_down_faces_local/average_down_faces_kernel below provide a
//      __half-clean drop-in for the REAL2 case only; REAL4/REAL8 continue
//      to use amrex::average_down_faces unchanged (float/double are
//      unaffected by the __half ambiguity).
//
// amrex::average_down_edges has only a MultiFab-specific overload (unlike
// average_down_nodal/average_down/average_down_faces, which are all
// `template<typename FAB>`), so schedule.cxx's Restrict cannot use it for
// edge-centered amrex::fMultiFab/hMultiFab (CCTK_REAL4/REAL2) groups. The
// function below is a drop-in FAB-templated replacement, structured as a
// line-by-line mirror of amrex::average_down_faces<FAB> (see
// AMReX_MultiFabUtil.H), with the direction test inverted (an edge index
// type is CELL-centered in exactly one direction and NODE-centered in the
// rest, the opposite of a face) and the point kernel replaced by
// average_down_edges_kernel below, a T-templated version of AMReX's own
// (Real-only) amrex_avgdown_edges point kernel (AMReX_MultiFabUtil_3D_C.H /
// AMReX_MultiFabUtil_nd_C.H): a coarse edge value is the unweighted average
// of the ratio[dir] fine edges it covers along the one cell-centered
// direction dir, exactly as AMReX's MultiFab-only average_down_edges does.
// (The GPU-fusing fast path that average_down_faces<FAB> additionally uses
// is a performance optimization, not a semantic difference, and is omitted
// here; AMREX_HOST_DEVICE_PARALLEL_FOR_4D below still dispatches correctly
// on GPU builds, just without kernel fusion.)
//
// average_down_faces_local/average_down_faces_kernel below are a second,
// independent mirror of the same amrex::average_down_faces<FAB> /
// amrex_avgdown_faces<T> pair (direction test: a face index type is
// NODE-centered in exactly one direction, same test as amrex's own), except
// the point kernel is written __half-clean from the start (see part (2)
// above), so it can be used for CCTK_REAL2 where amrex::average_down_faces
// itself cannot.

#include "driver.hxx"

#include <AMReX_MultiFabUtil.H>

namespace CarpetX {

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void average_down_edges_kernel(
    int i, int j, int k, int n, const amrex::Array4<T> &crse,
    const amrex::Array4<T const> &fine, const amrex::IntVect &ratio,
    int dir) noexcept {
  const int facx = ratio[0];
  const int facy = ratio[1];
  const int facz = ratio[2];
  const int ii = i * facx;
  const int jj = j * facy;
  const int kk = k * facz;
  switch (dir) {
  case 0: {
    const T facinv = T(1) / T(facx);
    T c(0);
    for (int iref = 0; iref < facx; ++iref)
      c += fine(ii + iref, jj, kk, n);
    crse(i, j, k, n) = c * facinv;
    break;
  }
  case 1: {
    const T facinv = T(1) / T(facy);
    T c(0);
    for (int jref = 0; jref < facy; ++jref)
      c += fine(ii, jj + jref, kk, n);
    crse(i, j, k, n) = c * facinv;
    break;
  }
  case 2: {
    const T facinv = T(1) / T(facz);
    T c(0);
    for (int kref = 0; kref < facz; ++kref)
      c += fine(ii, jj, kk + kref, n);
    crse(i, j, k, n) = c * facinv;
    break;
  }
  default:
    break;
  }
}

// FAB-templated mirror of amrex::average_down_edges(const MultiFab&,
// MultiFab&, const IntVect&, int); see the file comment above.
template <typename FAB>
void average_down_edges(const amrex::FabArray<FAB> &fine,
                        amrex::FabArray<FAB> &crse, const amrex::IntVect &ratio,
                        int ngcrse = 0) {
  using T = typename FAB::value_type;

  AMREX_ASSERT(crse.nComp() == fine.nComp());
  AMREX_ASSERT(fine.ixType() == crse.ixType());
  const auto type = fine.ixType();
  int dir;
  for (dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    if (!type.nodeCentered(dir))
      break;
  }
  auto tmptype = type;
  tmptype.set(dir);
  if (dir >= AMREX_SPACEDIM || !tmptype.nodeCentered()) {
    amrex::Abort("CarpetX::average_down_edges: not edge index type");
  }
  const int ncomp = crse.nComp();
  if (amrex::isMFIterSafe(fine, crse)) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(crse, amrex::TilingIfNotGPU()); mfi.isValid();
        ++mfi) {
      const amrex::Box &bx = mfi.growntilebox(ngcrse);
      auto const &crsearr = crse.array(mfi);
      auto const &finearr = fine.const_array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n, {
        average_down_edges_kernel<T>(i, j, k, n, crsearr, finearr, ratio,
                                     dir);
      });
    }
  } else {
    amrex::FabArray<FAB> ctmp(amrex::coarsen(fine.boxArray(), ratio),
                              fine.DistributionMap(), ncomp, ngcrse,
                              amrex::MFInfo(), amrex::DefaultFabFactory<FAB>());
    average_down_edges(fine, ctmp, ratio, ngcrse);
    crse.ParallelCopy(ctmp, 0, 0, ncomp, ngcrse, ngcrse);
  }
}

// __half-clean point kernel, mirroring amrex::amrex_avgdown_faces<T>
// (AMReX_MultiFabUtil_3D_C.H) exactly except that every division operand is
// forced to T so nothing mixes T with a bare int (see the file comment
// above, part (2)).
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void average_down_faces_kernel(
    int i, int j, int k, int n, const amrex::Array4<T> &crse,
    const amrex::Array4<T const> &fine, const amrex::IntVect &ratio,
    int idir) noexcept {
  const int facx = ratio[0];
  const int facy = ratio[1];
  const int facz = ratio[2];
  const int ii = i * facx;
  const int jj = j * facy;
  const int kk = k * facz;
  switch (idir) {
  case 0: {
    const T facinv = T(1) / T(facy * facz);
    T c(0);
    for (int kref = 0; kref < facz; ++kref)
      for (int jref = 0; jref < facy; ++jref)
        c += fine(ii, jj + jref, kk + kref, n);
    crse(i, j, k, n) = c * facinv;
    break;
  }
  case 1: {
    const T facinv = T(1) / T(facx * facz);
    T c(0);
    for (int kref = 0; kref < facz; ++kref)
      for (int iref = 0; iref < facx; ++iref)
        c += fine(ii + iref, jj, kk + kref, n);
    crse(i, j, k, n) = c * facinv;
    break;
  }
  case 2: {
    const T facinv = T(1) / T(facx * facy);
    T c(0);
    for (int jref = 0; jref < facy; ++jref)
      for (int iref = 0; iref < facx; ++iref)
        c += fine(ii + iref, jj + jref, kk, n);
    crse(i, j, k, n) = c * facinv;
    break;
  }
  default:
    break;
  }
}

// FAB-templated, __half-clean mirror of amrex::average_down_faces(const
// FabArray<FAB>&, FabArray<FAB>&, const IntVect&, int); see the file
// comment above (part (2)). Used only for CCTK_REAL2 (hMultiFab); REAL4/
// REAL8 keep using amrex::average_down_faces, which is fine for them.
template <typename FAB>
void average_down_faces_local(const amrex::FabArray<FAB> &fine,
                              amrex::FabArray<FAB> &crse,
                              const amrex::IntVect &ratio, int ngcrse = 0) {
  using T = typename FAB::value_type;

  AMREX_ASSERT(crse.nComp() == fine.nComp());
  AMREX_ASSERT(fine.ixType() == crse.ixType());
  const auto type = fine.ixType();
  int dir;
  for (dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    if (type.nodeCentered(dir))
      break;
  }
  auto tmptype = type;
  tmptype.unset(dir);
  if (dir >= AMREX_SPACEDIM || !tmptype.cellCentered()) {
    amrex::Abort("CarpetX::average_down_faces_local: not face index type");
  }
  const int ncomp = crse.nComp();
  if (amrex::isMFIterSafe(fine, crse)) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(crse, amrex::TilingIfNotGPU()); mfi.isValid();
        ++mfi) {
      const amrex::Box &bx = mfi.growntilebox(ngcrse);
      auto const &crsearr = crse.array(mfi);
      auto const &finearr = fine.const_array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n, {
        average_down_faces_kernel<T>(i, j, k, n, crsearr, finearr, ratio,
                                     dir);
      });
    }
  } else {
    amrex::FabArray<FAB> ctmp(amrex::coarsen(fine.boxArray(), ratio),
                              fine.DistributionMap(), ncomp, ngcrse,
                              amrex::MFInfo(), amrex::DefaultFabFactory<FAB>());
    average_down_faces_local(fine, ctmp, ratio, ngcrse);
    crse.ParallelCopy(ctmp, 0, 0, ncomp, ngcrse, ngcrse);
  }
}

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_RESTRICT_EDGES_HXX
