#include "fillpatch.hxx"
#include "schedule.hxx"

#include <utility>

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Version.H>

namespace CarpetX {

using namespace amrex;
#if AMREX_RELEASE_NUMBER >= 240500
using namespace amrex::detail;
#endif

// The code in this file is written in a "coroutine style"
// <https://en.wikipedia.org/wiki/Coroutine>. That is, each function
// returns another function that describes what to do next. This
// allows the caller to interleave many function calls, for example to
// schedule many calls to `MPI_Irecv` and `MPI_Isend` simultaneously.
//
// This programming style is obviously quite tedious. C++20 will have
// special support for this via `co_yield` etc.
// <https://en.cppreference.com/w/cpp/coroutine>, and the functions in
// this file will then look like normal functions.
//
// Coroutines were popularized in the "Modula" language in the 1980s.
// Welcome to the future, C++, you're only 40 years behind.

// Precision-generic replacement for amrex::detail::FillPatchInterp's
// duck-typed template<MF, Interp> overload (AMReX_FillPatchUtil_I.H,
// `FillPatchInterp(MF& mf_fine_patch, ..., Interp* mapper, ...)`), taking an
// InterpolaterT<T>* mapper (T = MF::value_type) instead of an
// amrex::Interpolater*/MFInterpolater*. CarpetX cannot call AMReX's own
// overload here: that overload's InterpBase* runtime-dispatch fallback only
// ever downcasts to amrex::Interpolater or amrex::MFInterpolater, neither of
// which InterpolaterT<CCTK_REAL4> is or could be (it operates on
// amrex::BaseFab<float>, which amrex::Interpolater's FArrayBox-typed
// interface cannot represent) -- so for CCTK_REAL4 groups AMReX's
// FillPatchInterp would have no matching overload at all. This function
// replicates the exact loop structure and semantics of that AMReX overload
// (including the amrex::setBC() call establishing per-call boundary types)
// for both precisions.
template <typename MF>
static void
fill_patch_interp(MF &mf_fine_patch, int fcomp, const MF &mf_crse_patch,
                  int ccomp, int ncomp, const IntVect &ng,
                  const Geometry &cgeom, const Geometry &fgeom,
                  const Box &dest_domain, const IntVect &ratio,
                  InterpolaterT<typename MF::value_type> *const mapper,
                  const Vector<BCRec> &bcs, int bcscomp) {
  const Box &cdomain = amrex::convert(cgeom.Domain(), mf_fine_patch.ixType());
  const int idummy = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  {
    Vector<BCRec> bcr(ncomp);
    for (MFIter mfi(mf_fine_patch); mfi.isValid(); ++mfi) {
      const auto &sfab = mf_crse_patch[mfi];
      const Box &sbx = sfab.box();

      auto &dfab = mf_fine_patch[mfi];
      const Box &dbx = amrex::grow(mfi.validbox(), ng) & dest_domain;

      amrex::setBC(sbx, cdomain, bcscomp, 0, ncomp, bcs, bcr);
      mapper->interp(sfab, ccomp, dfab, fcomp, ncomp, dbx, ratio, cgeom, fgeom,
                     bcr, idummy, idummy, RunOn::Gpu);
    }
  }
}

template <typename MF>
void FillPatch_Sync(task_manager &tasks2,
                    const GHExt::PatchData::LevelData::GroupData &groupdata,
                    MF &mfab, const Geometry &geom) {
  mfab.FillBoundary_nowait(0, mfab.nComp(), mfab.nGrowVect(),
                           geom.periodicity());
  tasks2.submit_serially([&groupdata, &mfab]() {
    mfab.FillBoundary_finish();
    groupdata.apply_boundary_conditions(mfab);
  });
}

template <typename MF>
void FillPatch_ProlongateGhosts(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata, MF &mfab,
    const MF &cmfab, const Geometry &fgeom, const Geometry &cgeom,
    InterpolaterT<typename MF::value_type> *const mapper,
    const Vector<BCRec> &bcrecs) {
  const IntVect &nghosts = mfab.nGrowVect();
  if (nghosts.max() == 0)
    return;

  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  const EB2::IndexSpace *const index_space = nullptr;

  const InterpolaterBoxCoarsener &coarsener = mapper->BoxCoarsener(ratio);

  const FabArrayBase::FPinfo &fpc = FabArrayBase::TheFPinfo(
      mfab, mfab, nghosts, coarsener, fgeom, cgeom, index_space);

  // Synchronize
  mfab.FillBoundary_nowait(0, mfab.nComp(), mfab.nGrowVect(),
                           fgeom.periodicity());

  if (fpc.ba_crse_patch.empty()) {
    // There is no coarser level for our boundaries, i.e. there is no
    // prolongation. Apply the boundary conditions right away.

    tasks2.submit_serially([&groupdata, &mfab]() {
      // Finish synchronizing
      mfab.FillBoundary_finish();

      // Apply symmetry and boundary conditions
      groupdata.apply_boundary_conditions(mfab);
    });
    return;
  }

  // Prolongate from the next coarser level. Apply the boundary
  // conditions after the prolongation is done (because symmetry
  // boundary conditions might require prolongated points).

  // Copy parts of coarse grid into temporary buffer
  MF *const mfab_crse_patch_ptr = new MF(make_mf_crse_patch<MF>(fpc, ncomps));
  MF &mfab_crse_patch = *mfab_crse_patch_ptr;
  mf_set_domain_bndry(mfab_crse_patch, cgeom);

  // This is not local
  mfab_crse_patch.ParallelCopy_nowait(
      cmfab, 0, 0, ncomps, IntVect{0} /* don't use coarse ghosts */,
      mfab_crse_patch.nGrowVect(), cgeom.periodicity());

  tasks2.submit_serially([&tasks3, &groupdata, &coarsegroupdata, &mfab, &cgeom,
                          &fgeom, mapper, &bcrecs, &fpc,
                          mfab_crse_patch_ptr]() {
    const IntVect &nghosts = mfab.nGrowVect();
    const int ncomps = mfab.nComp();
    const IntVect ratio{2, 2, 2};
    MF &mfab_crse_patch = *mfab_crse_patch_ptr;

    // Finish synchronizing
    mfab.FillBoundary_finish();

    // Finish copying parts of coarse grid into temporary buffer
    mfab_crse_patch.ParallelCopy_finish();

    coarsegroupdata.apply_boundary_conditions(mfab_crse_patch);

    MF *const mfab_fine_patch_ptr =
        new MF(make_mf_fine_patch<MF>(fpc, ncomps));
    MF &mfab_fine_patch = *mfab_fine_patch_ptr;

    // Interpolate coarse buffer into fine buffer (in space, local)
    fill_patch_interp(mfab_fine_patch, 0, mfab_crse_patch, 0, ncomps,
                      IntVect{0} /* don't add any new ghosts */, cgeom, fgeom,
                      grow(convert(fgeom.Domain(), mfab.ixType()), nghosts),
                      ratio, mapper, bcrecs, 0);

    // Copy fine buffer into destination
    mfab.ParallelCopy_nowait(
        mfab_fine_patch, 0, 0, ncomps,
        IntVect{0} /* don't use any ghosts from the buffer */, nghosts);

    delete mfab_crse_patch_ptr;

    tasks3.submit_serially([&groupdata, &mfab, mfab_fine_patch_ptr]() {
      // Finish copying fine buffer into destination
      mfab.ParallelCopy_finish();

      // Apply symmetry and boundary conditions
      groupdata.apply_boundary_conditions(mfab);

      delete mfab_fine_patch_ptr;
    });
  });
}

template <typename MF>
void FillPatch_NewLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata, MF &mfab,
    const MF &cmfab, const Geometry &cgeom, const Geometry &fgeom,
    InterpolaterT<typename MF::value_type> *const mapper,
    const Vector<BCRec> &bcrecs) {
  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  const IntVect &nghosts = mfab.nGrowVect();
  // const EB2::IndexSpace *const index_space = nullptr;

  const InterpolaterBoxCoarsener &coarsener = mapper->BoxCoarsener(ratio);

  const BoxArray &ba = mfab.boxArray();
  const DistributionMapping &dm = mfab.DistributionMap();

  const IndexType &ixtype = ba.ixType();
  assert(ixtype == cmfab.boxArray().ixType());

  // Suffix `_g` is for "with ghosts added"
  Box fdomain_g(amrex::convert(fgeom.Domain(), mfab.ixType()));
  for (int d = 0; d < dim; ++d)
    if (fgeom.isPeriodic(d))
      fdomain_g.grow(d, nghosts[d]);

  const int nboxes = ba.size();
  BoxArray cba_g(nboxes);
  for (int i = 0; i < nboxes; ++i) {
    Box box = amrex::convert(amrex::grow(ba[i], nghosts), ixtype);
    box &= fdomain_g;
    cba_g.set(i, coarsener.doit(box));
  }
  MF cmfab_g(cba_g, dm, ncomps, 0);
  mf_set_domain_bndry(cmfab_g, cgeom);

  cmfab_g.ParallelCopy(cmfab, 0, 0, ncomps, cgeom.periodicity());

  coarsegroupdata.apply_boundary_conditions(cmfab_g);

  fill_patch_interp(mfab, 0, cmfab_g, 0, ncomps, nghosts, cgeom, fgeom,
                    fdomain_g, ratio, mapper, bcrecs, 0);

  groupdata.apply_boundary_conditions(mfab);
}

template <typename MF>
void FillPatch_RemakeLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata, MF &mfab,
    const MF &cmfab, const MF &fmfab, const Geometry &cgeom,
    const Geometry &fgeom, InterpolaterT<typename MF::value_type> *const mapper,
    const Vector<BCRec> &bcrecs) {
  const int ncomps = mfab.nComp();
  const IntVect ratio{2, 2, 2};
  const IntVect &nghosts = mfab.nGrowVect();
  const EB2::IndexSpace *const index_space = nullptr;

  const InterpolaterBoxCoarsener &coarsener = mapper->BoxCoarsener(ratio);

  const FabArrayBase::FPinfo &fpc = FabArrayBase::TheFPinfo(
      fmfab, mfab, nghosts, coarsener, fgeom, cgeom, index_space);

  if (!fpc.ba_crse_patch.empty()) {
    MF mfab_crse_patch = make_mf_crse_patch<MF>(fpc, ncomps);
    mf_set_domain_bndry(mfab_crse_patch, cgeom);

    mfab_crse_patch.ParallelCopy(
        cmfab, 0, 0, ncomps, IntVect{0} /* don't use coarse ghosts */,
        mfab_crse_patch.nGrowVect(), cgeom.periodicity());
    coarsegroupdata.apply_boundary_conditions(mfab_crse_patch);

    MF mfab_fine_patch = make_mf_fine_patch<MF>(fpc, ncomps);

    // In space, local
    fill_patch_interp(mfab_fine_patch, 0, mfab_crse_patch, 0, ncomps,
                      IntVect{0} /* don't add any new ghosts */, cgeom, fgeom,
                      grow(convert(fgeom.Domain(), mfab.ixType()), nghosts),
                      ratio, mapper, bcrecs, 0);

    mfab.ParallelCopy_nowait(
        mfab_fine_patch, 0, 0, ncomps,
        IntVect{0} /* don't use any ghosts from the buffer */, nghosts);
    mfab.ParallelCopy_finish();
  }

  mfab.ParallelCopy(fmfab, 0, 0, ncomps, IntVect{0} /* don't use old ghosts */,
                    nghosts, fgeom.periodicity());
  groupdata.apply_boundary_conditions(mfab);
}

template void
FillPatch_Sync<MultiFab>(task_manager &,
                         const GHExt::PatchData::LevelData::GroupData &,
                         MultiFab &, const Geometry &);
template void
FillPatch_Sync<fMultiFab>(task_manager &,
                          const GHExt::PatchData::LevelData::GroupData &,
                          fMultiFab &, const Geometry &);

template void FillPatch_ProlongateGhosts<MultiFab>(
    task_manager &, task_manager &,
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, MultiFab &,
    const MultiFab &, const Geometry &, const Geometry &,
    InterpolaterT<CCTK_REAL> *, const Vector<BCRec> &);
template void FillPatch_ProlongateGhosts<fMultiFab>(
    task_manager &, task_manager &,
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, fMultiFab &,
    const fMultiFab &, const Geometry &, const Geometry &,
    InterpolaterT<CCTK_REAL4> *, const Vector<BCRec> &);

template void FillPatch_NewLevel<MultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, MultiFab &,
    const MultiFab &, const Geometry &, const Geometry &,
    InterpolaterT<CCTK_REAL> *, const Vector<BCRec> &);
template void FillPatch_NewLevel<fMultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, fMultiFab &,
    const fMultiFab &, const Geometry &, const Geometry &,
    InterpolaterT<CCTK_REAL4> *, const Vector<BCRec> &);

template void FillPatch_RemakeLevel<MultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, MultiFab &,
    const MultiFab &, const MultiFab &, const Geometry &, const Geometry &,
    InterpolaterT<CCTK_REAL> *, const Vector<BCRec> &);
template void FillPatch_RemakeLevel<fMultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, fMultiFab &,
    const fMultiFab &, const fMultiFab &, const Geometry &, const Geometry &,
    InterpolaterT<CCTK_REAL4> *, const Vector<BCRec> &);

} // namespace CarpetX
