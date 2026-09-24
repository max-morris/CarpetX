#include "driver.hxx"
#include "mpi_types.hxx"
#include "reduction.hxx"
#include "schedule.hxx"

#include <cctk_Parameters.h>

#include <AMReX_MultiFabUtil.H>
#include <AMReX_Orientation.H>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <variant>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CarpetX {

template <typename T, int D> MPI_Datatype reduction_mpi_datatype() {
  static MPI_Datatype datatype = MPI_DATATYPE_NULL;
  if (datatype == MPI_DATATYPE_NULL) {
    MPI_Type_contiguous(sizeof(reduction<T, D>) / sizeof(T),
                        mpi_datatype<T>::value, &datatype);
    char name[MPI_MAX_OBJECT_NAME];
    int namelen;
    MPI_Type_get_name(mpi_datatype<T>::value, name, &namelen);
    std::ostringstream buf;
    buf << "reduction<" << name << "," << D << ">";
    std::string newname = buf.str();
    MPI_Type_set_name(datatype, newname.c_str());
    MPI_Type_commit(&datatype);
  }
  return datatype;
}

namespace {
template <typename T>
void mpi_reduce_typed(const void *restrict x0, void *restrict y0,
                      const int *restrict length) {
  const T *restrict x = static_cast<const T *>(x0);
  T *restrict y = static_cast<T *>(y0);
#pragma omp simd
  for (int i = 0; i < *length; ++i)
    y[i] += x[i];
}

void mpi_reduce(void *restrict x, void *restrict y, int *restrict length,
                MPI_Datatype *restrict datatype) {
  // Analyze MPI datatype
  int num_integers, num_addresses, num_datatypes, combiner;
  MPI_Type_get_envelope(*datatype, &num_integers, &num_addresses,
                        &num_datatypes, &combiner);
  assert(combiner == MPI_COMBINER_CONTIGUOUS);
  assert(num_integers == 1);
  assert(num_datatypes == 1);
  std::vector<int> integers(num_integers);
  std::vector<MPI_Aint> addresses(num_addresses);
  std::vector<MPI_Datatype> datatypes(num_datatypes);
  MPI_Type_get_contents(*datatype, num_integers, num_addresses, num_datatypes,
                        integers.data(), addresses.data(), datatypes.data());
  MPI_Datatype inner_datatype = datatypes.at(0);
  if (inner_datatype == MPI_FLOAT)
    return mpi_reduce_typed<reduction<float, dim> >(x, y, length);
  if (inner_datatype == MPI_DOUBLE)
    return mpi_reduce_typed<reduction<double, dim> >(x, y, length);
  if (inner_datatype == MPI_LONG_DOUBLE)
    return mpi_reduce_typed<reduction<long double, dim> >(x, y, length);
  CCTK_ERROR("Unsupported MPI datatype");
}
} // namespace

MPI_Op reduction_mpi_op() {
  static MPI_Op op = MPI_OP_NULL;
  if (op == MPI_OP_NULL)
    MPI_Op_create(mpi_reduce, 1 /*commutes*/, &op);
  return op;
}

////////////////////////////////////////////////////////////////////////////////

namespace {
// Templated on the *source* array element type SrcT (CCTK_REAL for
// REAL/REAL8 groups, CCTK_REAL4 for CCTK_REAL4 groups, CCTK_REAL2 for
// CCTK_REAL2 groups). The accumulator always uses CCTK_REAL, i.e. CCTK_REAL4
// / CCTK_REAL2 source data is read and summed in double precision; the
// reduction result type is therefore unchanged by the group's storage
// precision.
template <typename SrcT>
reduction<CCTK_REAL, dim>
reduce_array(const amrex::Array4<const SrcT> &restrict vars, const int n,
             const vect<int, dim> &tmin, const vect<int, dim> &tmax,
             const vect<int, dim> &indextype, const vect<int, dim> &imin,
             const vect<int, dim> &imax,
             const amrex::Array4<const int> *restrict const finemask,
             const vect<CCTK_REAL, dim> &x0, const vect<CCTK_REAL, dim> &dx) {
  constexpr vect<vect<int, dim>, dim> di = {vect<int, dim>::unit(0),
                                            vect<int, dim>::unit(1),
                                            vect<int, dim>::unit(2)};
  constexpr vect<vect<vect<int, dim>, dim>, 2> dirs = {-di, +di};

  constexpr vect<vect<std::bitset<8>, dim>, 2> faces = {
      {0b01010101, 0b00110011, 0b00001111},
      {0b10101010, 0b11001100, 0b11110000}};

  const vect<vect<int, dim>, 2> ibnd = {imin, imax - 1};

  const auto masked = [&](const vect<int, dim> &ipos) {
    return finemask && (*finemask)(ipos[0], ipos[1], ipos[2]);
  };

  const CCTK_REAL dV = prod(dx);

  // Use per-loop reduction objects to reduce round-off error
  reduction<CCTK_REAL, dim> redk;
  // TODO: use loop.hxx code to loop over grid
  for (int k = tmin[2]; k < tmax[2]; ++k) {
    reduction<CCTK_REAL, dim> redj;
    for (int j = tmin[1]; j < tmax[1]; ++j) {
      reduction<CCTK_REAL, dim> redi;
      for (int i = tmin[0]; i < tmax[0]; ++i) {
        const vect<int, dim> ipos = {i, j, k};

        // For vertex-centred grids, ensure that points at the outer boundary
        // are counted with a weight of 1/2
        std::bitset<8> outer_active = 0b11111111;
        for (int f = 0; f < 2; ++f)
          for (int d = 0; d < dim; ++d)
            if (indextype[d] == 0 && ipos[d] == ibnd[f][d])
              outer_active &= ~faces[f][d];

        // For vertex-centred grids, ensure that points at refinement boundaries
        // are counted with a weight of 1/2
        std::bitset<8> inner_active;
        if (!masked(ipos)) {
          inner_active = 0b11111111;
        } else {
          inner_active = 0b00000000;
          for (int f = 0; f < 2; ++f)
            for (int d = 0; d < dim; ++d)
              if (indextype[d] == 0 &&
                  !(ipos[d] != ibnd[f][d] && masked(ipos + dirs[f][d])))
                inner_active |= faces[f][d];
        }

        assert((~outer_active & ~inner_active).none());

        const std::bitset<8> active = outer_active & inner_active;
        if (active.any()) {
          const CCTK_REAL W = active.count() / CCTK_REAL(active.size());

          const vect<CCTK_REAL, dim> x = x0 + ipos * dx;
          redi += reduction<CCTK_REAL, dim>(x, W * dV,
                                            CCTK_REAL(vars(i, j, k, n)));
        }
      }
      redj += redi;
    }
    redk += redj;
  }
  return redk;
}

// Templated on the AMReX FabArray specialization MF (amrex::MultiFab for
// REAL/REAL8 groups, amrex::fMultiFab for CCTK_REAL4 groups, hMultiFab for
// CCTK_REAL2 groups; same pattern as fillpatch.hxx's FillPatch_* functions).
// The source element type is `typename MF::value_type`; the reduction result
// is always accumulated (and MPI-reduced) in CCTK_REAL, so this returns the
// same result type for all storage precisions.
template <typename MF>
reduction<CCTK_REAL, dim> reduce_typed(int gi, int vi, int tl) {
  using SrcT = typename MF::value_type;

  reduction<CCTK_REAL, dim> red;
  // TODO: Parallelize over patches and levels
  for (auto &restrict patchdata : ghext->patchdata) {
    for (auto &restrict leveldata : patchdata.leveldata) {
      const auto &restrict groupdata = *leveldata.groupdata.at(gi);
      const MF &mfab = std::get<MF>(*groupdata.mfab.at(tl));
      std::unique_ptr<amrex::iMultiFab> finemask_imfab;

      warn_if_invalid(groupdata, vi, tl, make_valid_int(),
                      []() { return "Before reduction"; });

      const vect<int, dim> indextype = groupdata.indextype;

      const auto &restrict geom = patchdata.amrcore->Geom(leveldata.level);
      const CCTK_REAL *restrict const x01 = geom.ProbLo();
      const CCTK_REAL *restrict const dx1 = geom.CellSize();
      const vect<CCTK_REAL, dim> dx = {dx1[0], dx1[1], dx1[2]};
      const vect<CCTK_REAL, dim> x0v = {x01[0], x01[1], x01[2]};
      const auto x0 = x0v + indextype * dx / 2;

      const int fine_level = leveldata.level + 1;
      if (fine_level < int(patchdata.leveldata.size())) {
        const auto &restrict fine_leveldata =
            patchdata.leveldata.at(fine_level);
        const auto &restrict fine_groupdata = *fine_leveldata.groupdata.at(gi);
        // Only the box array is needed; the fine level may hold another
        // storage type than this one
        const amrex::BoxArray fine_ba = std::visit(
            [](const auto &fine_mfab) { return fine_mfab.boxArray(); },
            *fine_groupdata.mfab.at(tl));

        const amrex::IntVect reffact{2, 2, 2};

        finemask_imfab = std::make_unique<amrex::iMultiFab>(makeFineMask(
            mfab, fine_ba, reffact, geom.periodicity(),
            /*coarse value*/ 0, /* fine value */ 1));
      }

      auto mfitinfo = amrex::MFItInfo().SetDynamic(true).EnableTiling();
      // TODO: check that multi-threading actually helps (and we are
      // not dominated by memory latency)
      // TODO: document required version of OpenMP to use custom reductions
#ifdef __NVCOMPILER
#pragma omp parallel
      {
        auto &outer = red;
        reduction<CCTK_REAL, dim> red;
#else
#pragma omp parallel reduction(reduction : red)
#endif
        for (amrex::MFIter mfi(mfab, mfitinfo); mfi.isValid(); ++mfi) {
          const amrex::Box &bx = mfi.tilebox(); // current tile (without ghosts)
          const vect<int, dim> tmin{bx.smallEnd(0), bx.smallEnd(1),
                                    bx.smallEnd(2)};
          const vect<int, dim> tmax{bx.bigEnd(0) + 1, bx.bigEnd(1) + 1,
                                    bx.bigEnd(2) + 1};
          const amrex::Box &vbx =
              mfi.validbox(); // interior region (without ghosts)
          const vect<int, dim> imin{vbx.smallEnd(0), vbx.smallEnd(1),
                                    vbx.smallEnd(2)};
          const vect<int, dim> imax{vbx.bigEnd(0) + 1, vbx.bigEnd(1) + 1,
                                    vbx.bigEnd(2) + 1};

          const amrex::Array4<const SrcT> &vars = mfab.array(mfi);

          std::unique_ptr<amrex::Array4<const int> > finemask;
          if (finemask_imfab) {
            finemask = std::make_unique<amrex::Array4<const int> >(
                finemask_imfab->array(mfi));
            // Ensure the mask has the correct size
            assert(finemask->begin.x == vars.begin.x);
            assert(finemask->begin.y == vars.begin.y);
            assert(finemask->begin.z == vars.begin.z);
            assert(finemask->end.x == vars.end.x);
            assert(finemask->end.y == vars.end.y);
            assert(finemask->end.z == vars.end.z);
          }

          red += reduce_array<SrcT>(vars, vi, tmin, tmax, indextype, imin,
                                    imax, finemask.get(), x0, dx);
        }
#ifdef __NVCOMPILER
#pragma omp critical (CarpetX_reduce)
        outer += red;
      }
#endif
    }
  }

  MPI_Datatype datatype = reduction_mpi_datatype<CCTK_REAL, dim>();
  MPI_Op op = reduction_mpi_op();
  MPI_Allreduce(MPI_IN_PLACE, &red, 1, datatype, op, MPI_COMM_WORLD);

  return red;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Per-level norms and fp16 admissibility (out_norm_per_level)

namespace {

// Thread-local accumulator; merged under a critical section (no custom
// OpenMP reduction, unlike reduce_typed above).
struct level_acc_t {
  CCTK_REAL sum2 = 0;
  CCTK_REAL maxabs = 0;
  CCTK_REAL min = +1.0 / 0.0;
  CCTK_REAL max = -1.0 / 0.0;
  long long npoints = 0;
  long long nsubnormal16 = 0;
  CCTK_REAL admiss = -1.0 / 0.0;
  CCTK_REAL admiss_shell = -1.0 / 0.0;
  long long nnan = 0; // ratios that were NaN (stale ghost, poison); reported as NaN

  void merge(const level_acc_t &o) noexcept {
    sum2 += o.sum2;
    maxabs = std::max(maxabs, o.maxabs);
    min = std::min(min, o.min);
    max = std::max(max, o.max);
    npoints += o.npoints;
    nsubnormal16 += o.nsubnormal16;
    admiss = std::max(admiss, o.admiss);
    admiss_shell = std::max(admiss_shell, o.admiss_shell);
    nnan += o.nnan;
  }
};

// binary16's minimum normal, 2^-14
constexpr CCTK_REAL fp16_min_normal = 6.103515625e-5;

// Half-ulp of binary16 at magnitude |u|: 2^(floor(log2 |u|) - 11) in the
// normal range (10 fraction bits, so the ulp is 2^(e-10)), and half of the
// subnormal spacing 2^-24 below it (also for u == 0).
inline CCTK_REAL fp16_half_ulp(const CCTK_REAL absu) noexcept {
  using std::ilogb, std::ldexp;
  if (!(absu >= fp16_min_normal))
    return ldexp(CCTK_REAL(1), -25);
  return ldexp(CCTK_REAL(1), ilogb(absu) - 11);
}

// One box: pass 1 (norms over the interior, mask applied) and pass 2 (the
// admissibility ratio over the interior points not covered by a finer
// level, mask applied). `vars` and `mask` are the full arrays including
// ghost zones; [tmin, tmax) is the tile, a subset of the interior. Values
// are read in the storage type SrcT and converted to CCTK_REAL once per
// point; D2u is formed in SrcT (three loads, two subtractions in the
// storage type's own arithmetic) and converted afterwards, so the ratio
// judges the stored data, not a widened copy of it.
template <typename SrcT, typename MaskT>
void reduce_level_box(level_acc_t &acc,
                      const amrex::Array4<const SrcT> &restrict vars,
                      const int n, const amrex::Array4<const MaskT> *mask,
                      const int mask_n, const CCTK_REAL mask_below,
                      const vect<int, dim> &tmin,
                      const vect<int, dim> &tmax,
                      const amrex::Array4<const int> *restrict const finemask,
                      const bool have_ghosts, const vect<CCTK_REAL, dim> &x0,
                      const vect<CCTK_REAL, dim> &dx, const CCTK_REAL floor0,
                      const CCTK_REAL shell_radius,
                      const CCTK_REAL shell_halfwidth) {
  using std::fabs, std::sqrt;
  for (int k = tmin[2]; k < tmax[2]; ++k) {
    for (int j = tmin[1]; j < tmax[1]; ++j) {
      for (int i = tmin[0]; i < tmax[0]; ++i) {
        if (mask && CCTK_REAL((*mask)(i, j, k, mask_n)) < mask_below)
          continue;

        // Pass 1: norms
        const CCTK_REAL u = CCTK_REAL(vars(i, j, k, n));
        const CCTK_REAL absu = fabs(u);
        acc.sum2 += u * u;
        acc.maxabs = std::max(acc.maxabs, absu);
        acc.min = std::min(acc.min, u);
        acc.max = std::max(acc.max, u);
        ++acc.npoints;
        if (absu > 0 && absu < fp16_min_normal)
          ++acc.nsubnormal16;

        // Pass 2: admissibility, fine-masked
        if (!have_ghosts)
          continue;
        if (finemask && (*finemask)(i, j, k))
          continue;
        const SrcT u0 = vars(i, j, k, n);
        const SrcT d2x = (vars(i + 1, j, k, n) - u0) - (u0 - vars(i - 1, j, k, n));
        const SrcT d2y = (vars(i, j + 1, k, n) - u0) - (u0 - vars(i, j - 1, k, n));
        const SrcT d2z = (vars(i, j, k + 1, n) - u0) - (u0 - vars(i, j, k - 1, n));
        const CCTK_REAL D2 = std::max(
            {fabs(CCTK_REAL(d2x)), fabs(CCTK_REAL(d2y)), fabs(CCTK_REAL(d2z))});
        const vect<int, dim> ipos = {i, j, k};
        const vect<CCTK_REAL, dim> x = x0 + ipos * dx;
        const CCTK_REAL r = sqrt(sum(x * x));
        const CCTK_REAL floor = floor0 / std::max(r, CCTK_REAL(1));
        const CCTK_REAL ratio = fp16_half_ulp(absu) / std::max(D2, floor);
        if (!(ratio == ratio)) {
          // std::max would drop a NaN silently; count it instead
          ++acc.nnan;
          continue;
        }
        acc.admiss = std::max(acc.admiss, ratio);
        if (fabs(r - shell_radius) < shell_halfwidth)
          acc.admiss_shell = std::max(acc.admiss_shell, ratio);
      }
    }
  }
}

// One level of one patch, over every box, for a given source and mask
// storage type (both dispatched through std::visit on AnyMultiFab).
template <typename MF, typename MaskMF>
void reduce_level_patch(level_acc_t &acc, const GHExt::PatchData &patchdata,
                        const GHExt::PatchData::LevelData &leveldata,
                        const GHExt::PatchData::LevelData::GroupData &groupdata,
                        const MF &mfab, const int gi, const int vi,
                        const int tl, const MaskMF *const mask_mfab,
                        const int mask_n, const CCTK_REAL mask_below,
                        const bool have_ghosts, const CCTK_REAL floor0,
                        const CCTK_REAL shell_radius,
                        const CCTK_REAL shell_halfwidth) {
  using SrcT = typename MF::value_type;
  using MaskT = typename MaskMF::value_type;

  const vect<int, dim> indextype = groupdata.indextype;

  const auto &restrict geom = patchdata.amrcore->Geom(leveldata.level);
  const CCTK_REAL *restrict const x01 = geom.ProbLo();
  const CCTK_REAL *restrict const dx1 = geom.CellSize();
  const vect<CCTK_REAL, dim> dx = {dx1[0], dx1[1], dx1[2]};
  const vect<CCTK_REAL, dim> x0v = {x01[0], x01[1], x01[2]};
  const auto x0 = x0v + indextype * dx / 2;
  // floor(l, x) = amplitude * (radius / max(|x|, 1)) * dx_l^2; the
  // radius-dependent factor is applied per point in reduce_level_box.
  const CCTK_REAL floor0_l = floor0 * dx[0] * dx[0];

  std::unique_ptr<amrex::iMultiFab> finemask_imfab;
  const int fine_level = leveldata.level + 1;
  if (fine_level < int(patchdata.leveldata.size())) {
    const auto &restrict fine_leveldata = patchdata.leveldata.at(fine_level);
    const auto &restrict fine_groupdata = *fine_leveldata.groupdata.at(gi);
    // Only the fine box array is needed, and every AnyMultiFab alternative
    // provides it, so the fine level may hold another storage type than
    // this one (per-level widths).
    const amrex::BoxArray fine_ba = std::visit(
        [](const auto &fine_mfab) { return fine_mfab.boxArray(); },
        *fine_groupdata.mfab.at(tl));
    const amrex::IntVect reffact{2, 2, 2};
    finemask_imfab = std::make_unique<amrex::iMultiFab>(
        makeFineMask(mfab, fine_ba, reffact, geom.periodicity(),
                     /*coarse value*/ 0, /* fine value */ 1));
  }

  auto mfitinfo = amrex::MFItInfo().SetDynamic(true).EnableTiling();
#pragma omp parallel
  {
    level_acc_t local;
    for (amrex::MFIter mfi(mfab, mfitinfo); mfi.isValid(); ++mfi) {
      const amrex::Box &bx = mfi.tilebox(); // current tile (without ghosts)
      const vect<int, dim> tmin{bx.smallEnd(0), bx.smallEnd(1), bx.smallEnd(2)};
      const vect<int, dim> tmax{bx.bigEnd(0) + 1, bx.bigEnd(1) + 1,
                                bx.bigEnd(2) + 1};

      const amrex::Array4<const SrcT> &vars = mfab.array(mfi);

      std::unique_ptr<amrex::Array4<const int> > finemask;
      if (finemask_imfab) {
        finemask = std::make_unique<amrex::Array4<const int> >(
            finemask_imfab->array(mfi));
        assert(finemask->begin.x == vars.begin.x);
        assert(finemask->begin.y == vars.begin.y);
        assert(finemask->begin.z == vars.begin.z);
        assert(finemask->end.x == vars.end.x);
        assert(finemask->end.y == vars.end.y);
        assert(finemask->end.z == vars.end.z);
      }

      std::unique_ptr<amrex::Array4<const MaskT> > mask;
      if (mask_mfab)
        mask = std::make_unique<amrex::Array4<const MaskT> >(
            mask_mfab->array(mfi));

      reduce_level_box<SrcT, MaskT>(local, vars, vi, mask.get(), mask_n,
                                    mask_below, tmin, tmax, finemask.get(),
                                    have_ghosts,
                                    x0, dx, floor0_l, shell_radius,
                                    shell_halfwidth);
    }
#pragma omp critical(CarpetX_reduce_level)
    acc.merge(local);
  }
}

} // namespace

level_reduction_t reduce_level(const int gi, const int vi, const int tl,
                               const int level, const int mask_vi) {
  DECLARE_CCTK_PARAMETERS;

  cGroup group;
  int ierr = CCTK_GroupData(gi, &group);
  assert(!ierr);
  assert(group.grouptype == CCTK_GF);

  int mask_gi = -1, mask_vi0 = -1;
  if (mask_vi >= 0) {
    mask_gi = CCTK_GroupIndexFromVarI(mask_vi);
    mask_vi0 = mask_vi - CCTK_FirstVarIndexI(mask_gi);
    assert(mask_gi >= 0 && mask_vi0 >= 0);
  }

  const CCTK_REAL floor0 = out_norm_admiss_amplitude * out_norm_admiss_radius;

  level_acc_t acc;
  // The admissibility pass reads one point into the ghost and outer
  // boundary layers. It runs only when the group has ghost zones and those
  // layers are valid on every patch; otherwise only the norms pass runs and
  // the admissibility columns read NaN. Decided silently: groups written
  // interior-only with no SYNC (the constraint groups the gate reads) are
  // the normal case, not a warning.
  bool have_ghosts = true;
  for (const auto &restrict patchdata : ghext->patchdata) {
    if (level >= int(patchdata.leveldata.size()))
      continue;
    const auto &restrict groupdata =
        *patchdata.leveldata.at(level).groupdata.at(gi);
    const valid_t have = groupdata.valid.at(tl).at(vi).get();
    have_ghosts = have_ghosts && groupdata.nghostzones[0] >= 1 &&
                  groupdata.nghostzones[1] >= 1 &&
                  groupdata.nghostzones[2] >= 1 && have.valid_outer &&
                  have.valid_ghosts;
  }
  for (const auto &restrict patchdata : ghext->patchdata) {
    if (level >= int(patchdata.leveldata.size()))
      continue;
    const auto &restrict leveldata = patchdata.leveldata.at(level);
    const auto &restrict groupdata = *leveldata.groupdata.at(gi);

    warn_if_invalid(groupdata, vi, tl, make_valid_int(),
                    []() { return "Before per-level reduction"; });

    const GHExt::PatchData::LevelData::GroupData *mask_groupdata = nullptr;
    if (mask_gi >= 0) {
      mask_groupdata = leveldata.groupdata.at(mask_gi).get();
      if (mask_groupdata->indextype != groupdata.indextype) {
        // A mask of another centering cannot be applied point by point;
        // reduce this group unmasked and say so once per (group, level).
        static std::set<std::pair<int, int> > warned;
        if (warned.insert({gi, level}).second)
          CCTK_VWARN(CCTK_WARN_ALERT,
                     "out_norm_mask_var %s has a different centering than "
                     "group %s; level %d of that group is reduced unmasked",
                     CCTK_FullVarName(mask_vi), CCTK_FullGroupName(gi), level);
        mask_groupdata = nullptr;
      } else {
        warn_if_invalid(*mask_groupdata, mask_vi0, tl, make_valid_int(),
                        []() { return "Before per-level reduction (mask)"; });
      }
    }

    std::visit(
        [&](const auto &mfab) {
          using MF = std::decay_t<decltype(mfab)>;
          if (!mask_groupdata) {
            // No mask: pass a null pointer of the source type
            reduce_level_patch<MF, MF>(acc, patchdata, leveldata, groupdata,
                                       mfab, gi, vi, tl, nullptr, 0,
                                       out_norm_mask_below, have_ghosts, floor0,
                                       out_norm_admiss_radius,
                                       out_norm_admiss_shell_halfwidth);
            return;
          }
          std::visit(
              [&](const auto &mask_mfab) {
                using MaskMF = std::decay_t<decltype(mask_mfab)>;
                if (mask_mfab.boxArray() != mfab.boxArray() ||
                    mask_mfab.DistributionMap() != mfab.DistributionMap())
                  CCTK_VERROR("out_norm_mask_var %s is not laid out like the "
                              "reduced group %s on level %d",
                              CCTK_FullVarName(mask_vi), CCTK_FullGroupName(gi),
                              level);
                reduce_level_patch<MF, MaskMF>(
                    acc, patchdata, leveldata, groupdata, mfab, gi, vi, tl,
                    &mask_mfab, mask_vi0, out_norm_mask_below, have_ghosts,
                    floor0,
                    out_norm_admiss_radius, out_norm_admiss_shell_halfwidth);
              },
              *mask_groupdata->mfab.at(tl));
        },
        *groupdata.mfab.at(tl));
  }

  // MPI reduction: plain arrays, no custom datatype
  CCTK_REAL sums[1] = {acc.sum2};
  long long counts[3] = {acc.npoints, acc.nsubnormal16, acc.nnan};
  CCTK_REAL maxs[4] = {acc.maxabs, acc.max, acc.admiss, acc.admiss_shell};
  CCTK_REAL mins[1] = {acc.min};
  MPI_Allreduce(MPI_IN_PLACE, sums, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxs, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, mins, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  level_reduction_t red;
  red.sum2 = sums[0];
  red.maxabs = maxs[0];
  red.min = mins[0];
  red.max = maxs[1];
  red.npoints = counts[0];
  red.nsubnormal16 = counts[1];
  red.admiss = maxs[2];
  red.admiss_shell = maxs[3];
  red.has_admiss = have_ghosts;
  if (counts[2] > 0) {
    // A NaN ratio means a NaN in the data or its ghosts; do not hide it
    red.admiss = 0.0 / 0.0;
    red.admiss_shell = 0.0 / 0.0;
  }
  return red;
}

reduction<CCTK_REAL, dim> reduce(int gi, int vi, int tl) {
  DECLARE_CCTK_PARAMETERS;

  cGroup group;
  int ierr = CCTK_GroupData(gi, &group);
  assert(!ierr);
  assert(group.grouptype == CCTK_GF);

  // Dispatch on the group's storage precision. The source array element
  // type differs (CCTK_REAL4/CCTK_REAL2 vs. CCTK_REAL), but the accumulator
  // and MPI-reduced result always use CCTK_REAL (see reduce_typed above), so
  // the returned reduction<CCTK_REAL, dim> is identical in shape for all
  // precisions.
#ifdef HAVE_CCTK_REAL2
  if (vartype_is_real2(group.vartype))
    return reduce_typed<hMultiFab>(gi, vi, tl);
#endif
  if (vartype_is_real4(group.vartype))
    return reduce_typed<amrex::fMultiFab>(gi, vi, tl);
  return reduce_typed<amrex::MultiFab>(gi, vi, tl);
}

} // namespace CarpetX
