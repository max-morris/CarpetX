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
  // Accumulate in `compute_t<T>` (driver.hxx): for T=CCTK_REAL2 this widens
  // to float, avoiding the precision loss (and, for larger stencils,
  // overflow) of summing several _Float16 values directly. No-op for
  // REAL4/REAL8 (compute_t<T> == T there).
  using CT = compute_t<T>;
  const int facx = ratio[0];
  const int facy = ratio[1];
  const int facz = ratio[2];
  const int ii = i * facx;
  const int jj = j * facy;
  const int kk = k * facz;
  switch (dir) {
  case 0: {
    const CT facinv = CT(1) / CT(facx);
    CT c(0);
    for (int iref = 0; iref < facx; ++iref)
      c += CT(fine(ii + iref, jj, kk, n));
    crse(i, j, k, n) = T(c * facinv);
    break;
  }
  case 1: {
    const CT facinv = CT(1) / CT(facy);
    CT c(0);
    for (int jref = 0; jref < facy; ++jref)
      c += CT(fine(ii, jj + jref, kk, n));
    crse(i, j, k, n) = T(c * facinv);
    break;
  }
  case 2: {
    const CT facinv = CT(1) / CT(facz);
    CT c(0);
    for (int kref = 0; kref < facz; ++kref)
      c += CT(fine(ii, jj, kk + kref, n));
    crse(i, j, k, n) = T(c * facinv);
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
  // See average_down_edges_kernel above: accumulate in compute_t<T>
  // (float for CCTK_REAL2, T otherwise) instead of T.
  using CT = compute_t<T>;
  const int facx = ratio[0];
  const int facy = ratio[1];
  const int facz = ratio[2];
  const int ii = i * facx;
  const int jj = j * facy;
  const int kk = k * facz;
  switch (idir) {
  case 0: {
    const CT facinv = CT(1) / CT(facy * facz);
    CT c(0);
    for (int kref = 0; kref < facz; ++kref)
      for (int jref = 0; jref < facy; ++jref)
        c += CT(fine(ii, jj + jref, kk + kref, n));
    crse(i, j, k, n) = T(c * facinv);
    break;
  }
  case 1: {
    const CT facinv = CT(1) / CT(facx * facz);
    CT c(0);
    for (int kref = 0; kref < facz; ++kref)
      for (int iref = 0; iref < facx; ++iref)
        c += CT(fine(ii + iref, jj, kk + kref, n));
    crse(i, j, k, n) = T(c * facinv);
    break;
  }
  case 2: {
    const CT facinv = CT(1) / CT(facx * facy);
    CT c(0);
    for (int jref = 0; jref < facy; ++jref)
      for (int iref = 0; iref < facx; ++iref)
        c += CT(fine(ii + iref, jj + jref, kk, n));
    crse(i, j, k, n) = T(c * facinv);
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

// D5: FAB-templated mirror of amrex::average_down(const FabArray<FAB>&,
// FabArray<FAB>&, int scomp, int ncomp, const IntVect&) (AMReX_MultiFabUtil.H),
// restricted to the cell-centered (rank-3, "volume") case -- the only case
// this is used for -- with the accumulation done in `compute_t<T>`
// (driver.hxx) rather than T. amrex::average_down's own point kernel,
// amrex_avgdown (AMReX_MultiFabUtil_3D_C.H), sums `ratio[0]*ratio[1]*ratio[2]`
// fine values directly in T; for T=CCTK_REAL2 (_Float16, max finite value
// 65504) that overflows to +-infinity whenever the local mean of those
// values exceeds roughly 65504/(facx*facy*facz) (e.g. ~8192 for a factor-2
// 3-D refinement), and loses precision even when it does not. REAL4/REAL8
// keep using amrex::average_down unchanged (compute_t<T> == T there, and
// summing a handful of doubles/floats has none of that risk).
//
// Structured as a line-by-line mirror of amrex::average_down<FAB>'s
// non-nodal branch: same BoxArray/DistributionMap comparison to pick the
// direct path vs. the coarsen-into-a-temporary-then-ParallelCopy fallback,
// same tilebox and AMREX_HOST_DEVICE_PARALLEL_FOR_4D launch style as
// average_down_faces_local above. The GPU-fusing fast path
// (Gpu::inLaunchRegion() && isFusingCandidate()) is intentionally omitted,
// matching average_down_faces_local; AMREX_HOST_DEVICE_PARALLEL_FOR_4D below
// still dispatches correctly on GPU builds, just without kernel fusion.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void average_down_kernel(
    int i, int j, int k, int n, const amrex::Array4<T> &crse,
    const amrex::Array4<T const> &fine, int ccomp, int fcomp,
    const amrex::IntVect &ratio) noexcept {
  using CT = compute_t<T>;
  const int facx = ratio[0];
  const int facy = ratio[1];
  const int facz = ratio[2];
  const CT volfrac = CT(1) / CT(facx * facy * facz);
  const int ii = i * facx;
  const int jj = j * facy;
  const int kk = k * facz;
  CT c(0);
  for (int kref = 0; kref < facz; ++kref)
    for (int jref = 0; jref < facy; ++jref)
      for (int iref = 0; iref < facx; ++iref)
        c += CT(fine(ii + iref, jj + jref, kk + kref, n + fcomp));
  crse(i, j, k, n + ccomp) = T(volfrac * c);
}

template <typename FAB>
void average_down_local(const amrex::FabArray<FAB> &fine,
                        amrex::FabArray<FAB> &crse, int scomp, int ncomp,
                        const amrex::IntVect &ratio) {
  using T = typename FAB::value_type;

  AMREX_ASSERT(crse.nComp() >= scomp + ncomp);
  AMREX_ASSERT(fine.nComp() >= scomp + ncomp);
  AMREX_ASSERT(crse.is_cell_centered() && fine.is_cell_centered());

  amrex::BoxArray crse_fine_ba = fine.boxArray();
  crse_fine_ba.coarsen(ratio);

  if (crse_fine_ba == crse.boxArray() &&
      fine.DistributionMap() == crse.DistributionMap()) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(crse, amrex::TilingIfNotGPU()); mfi.isValid();
        ++mfi) {
      // NOTE: The tilebox is defined at the coarse level.
      const amrex::Box &bx = mfi.tilebox();
      auto const &crsearr = crse.array(mfi);
      auto const &finearr = fine.const_array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n, {
        average_down_kernel<T>(i, j, k, n, crsearr, finearr, scomp, scomp,
                               ratio);
      });
    }
  } else {
    amrex::FabArray<FAB> crse_fine(crse_fine_ba, fine.DistributionMap(),
                                   ncomp, 0, amrex::MFInfo(),
                                   amrex::DefaultFabFactory<FAB>());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(crse_fine, amrex::TilingIfNotGPU()); mfi.isValid();
        ++mfi) {
      const amrex::Box &bx = mfi.tilebox();
      auto const &crsearr = crse_fine.array(mfi);
      auto const &finearr = fine.const_array(mfi);
      AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n, {
        average_down_kernel<T>(i, j, k, n, crsearr, finearr, 0, scomp, ratio);
      });
    }
    crse.ParallelCopy(crse_fine, 0, scomp, ncomp);
  }
}

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_RESTRICT_EDGES_HXX
