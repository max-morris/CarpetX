#ifndef CARPETX_CARPETX_FILLPATCH_HXX
#define CARPETX_CARPETX_FILLPATCH_HXX

#include "driver.hxx"
#include "task_manager.hxx"

#include <functional>

namespace CarpetX {

// The four FillPatch_* functions below are templated on the AMReX
// FabArray specialization MF (amrex::MultiFab for REAL/REAL8 groups,
// amrex::fMultiFab for CCTK_REAL4 groups), so that they work uniformly for
// both storage precisions. The interpolator (`mapper`) is typed
// InterpolaterT<T>* with T = typename MF::value_type (CCTK_REAL for
// amrex::MultiFab, CCTK_REAL4 for amrex::fMultiFab; see interp_base.hxx and
// driver.hxx's GroupData::interpolator_real8/interpolator_real4).
//
// Both specializations (MF = amrex::MultiFab, amrex::fMultiFab) are
// explicitly instantiated in fillpatch.cxx.

// Sync
template <typename MF>
void FillPatch_Sync(task_manager &tasks2,
                    const GHExt::PatchData::LevelData::GroupData &groupdata,
                    MF &mfab, const amrex::Geometry &geom);

// Prolongate (but do not sync) ghosts. Expects coarse mfab synced (?)
// (but not necessarily ghost-prolongated).
template <typename MF>
void FillPatch_ProlongateGhosts(
    task_manager &tasks2, task_manager &tasks3,
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata, MF &mfab,
    const MF &cmfab, const amrex::Geometry &fgeom, const amrex::Geometry &cgeom,
    InterpolaterT<typename MF::value_type> *mapper,
    const amrex::Vector<amrex::BCRec> &bcrecs);

#warning "TODO: Restrict"

// Prolongate and sync interior. Expects coarse mfab prolongated and
// synced. ("InterpFromCoarseLevel")
template <typename MF>
void FillPatch_NewLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata, MF &mfab,
    const MF &cmfab, const amrex::Geometry &cgeom, const amrex::Geometry &fgeom,
    InterpolaterT<typename MF::value_type> *mapper,
    const amrex::Vector<amrex::BCRec> &bcrecs);

// ("FillPatchTwoLevels")
template <typename MF>
void FillPatch_RemakeLevel(
    const GHExt::PatchData::LevelData::GroupData &groupdata,
    const GHExt::PatchData::LevelData::GroupData &coarsegroupdata, MF &mfab,
    const MF &cmfab, const MF &fmfab, const amrex::Geometry &cgeom,
    const amrex::Geometry &fgeom, InterpolaterT<typename MF::value_type> *mapper,
    const amrex::Vector<amrex::BCRec> &bcrecs);

extern template void
FillPatch_Sync<amrex::MultiFab>(task_manager &,
                                const GHExt::PatchData::LevelData::GroupData &,
                                amrex::MultiFab &, const amrex::Geometry &);
extern template void
FillPatch_Sync<amrex::fMultiFab>(task_manager &,
                                 const GHExt::PatchData::LevelData::GroupData &,
                                 amrex::fMultiFab &, const amrex::Geometry &);

extern template void FillPatch_ProlongateGhosts<amrex::MultiFab>(
    task_manager &, task_manager &,
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, amrex::MultiFab &,
    const amrex::MultiFab &, const amrex::Geometry &, const amrex::Geometry &,
    InterpolaterT<CCTK_REAL> *, const amrex::Vector<amrex::BCRec> &);
extern template void FillPatch_ProlongateGhosts<amrex::fMultiFab>(
    task_manager &, task_manager &,
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, amrex::fMultiFab &,
    const amrex::fMultiFab &, const amrex::Geometry &, const amrex::Geometry &,
    InterpolaterT<CCTK_REAL4> *, const amrex::Vector<amrex::BCRec> &);

extern template void FillPatch_NewLevel<amrex::MultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, amrex::MultiFab &,
    const amrex::MultiFab &, const amrex::Geometry &, const amrex::Geometry &,
    InterpolaterT<CCTK_REAL> *, const amrex::Vector<amrex::BCRec> &);
extern template void FillPatch_NewLevel<amrex::fMultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, amrex::fMultiFab &,
    const amrex::fMultiFab &, const amrex::Geometry &, const amrex::Geometry &,
    InterpolaterT<CCTK_REAL4> *, const amrex::Vector<amrex::BCRec> &);

extern template void FillPatch_RemakeLevel<amrex::MultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, amrex::MultiFab &,
    const amrex::MultiFab &, const amrex::MultiFab &, const amrex::Geometry &,
    const amrex::Geometry &, InterpolaterT<CCTK_REAL> *,
    const amrex::Vector<amrex::BCRec> &);
extern template void FillPatch_RemakeLevel<amrex::fMultiFab>(
    const GHExt::PatchData::LevelData::GroupData &,
    const GHExt::PatchData::LevelData::GroupData &, amrex::fMultiFab &,
    const amrex::fMultiFab &, const amrex::fMultiFab &, const amrex::Geometry &,
    const amrex::Geometry &, InterpolaterT<CCTK_REAL4> *,
    const amrex::Vector<amrex::BCRec> &);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_FILLPATCH_HXX
