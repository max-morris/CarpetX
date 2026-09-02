#ifndef CARPETX_CARPETX_DRIVER_HXX
#define CARPETX_CARPETX_DRIVER_HXX

#include "interp_base.hxx"
#include "loop.hxx"
#include "mpi_typemap_real2.hxx"
#include "valid.hxx"

#include <rational.hxx>
#include <tuple.hxx>

#include <cctk.h>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <type_traits>
#include <variant>
#include <vector>

namespace CarpetX {
using namespace Arith;

using Loop::dim;

using rat64 = rational<int64_t>;

// TODO: It seems that AMReX now also has `RB90`, `RB180`, and
// `PolarB` boundary conditions. Make these available as well.

// Symmetries are domain properties
enum class symmetry_t {
  none,
  interpatch,
  periodic,
  reflection,
};
std::ostream &operator<<(std::ostream &os, const symmetry_t symmetry);

// Boundary conditions are group properties. They are valid only for faces where
// the domain symmetry is `none`.
enum class boundary_t {
  none,
  symmetry_boundary,
  dirichlet,
  linear_extrapolation,
  neumann,
  robin,
};
std::ostream &operator<<(std::ostream &os, const boundary_t boundary);

static_assert(AMREX_SPACEDIM == dim,
              "AMReX's AMREX_SPACEDIM must be the same as Cactus's cctk_dim");

static_assert(std::is_same<amrex::Real, CCTK_REAL>::value,
              "AMReX's Real type must be the same as Cactus's CCTK_REAL");

////////////////////////////////////////////////////////////////////////////////

// AMR driver
class CactusAmrCore final : public amrex::AmrCore {
  int patch;

public:
  bool cactus_is_initialized = false;
  std::vector<bool> level_modified;

  CactusAmrCore();
  CactusAmrCore(int patch, const amrex::RealBox *rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord = -1,
                amrex::Vector<amrex::IntVect> ref_ratios =
                    amrex::Vector<amrex::IntVect>(),
                const int *is_per = nullptr);
  CactusAmrCore(int patch, const amrex::RealBox &rb, int max_level_in,
                const amrex::Vector<int> &n_cell_in, int coord,
                amrex::Vector<amrex::IntVect> const &ref_ratios,
                amrex::Array<int, AMREX_SPACEDIM> const &is_per);
  CactusAmrCore(const amrex::AmrCore &rhs) = delete;
  CactusAmrCore &operator=(const amrex::AmrCore &rhs) = delete;

  virtual ~CactusAmrCore() override;

  virtual void ErrorEst(int level, amrex::TagBoxArray &tags, amrex::Real time,
                        int ngrow) override;
  void SetupLevel(int level, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);
  virtual void
  MakeNewLevelFromScratch(int level, amrex::Real time,
                          const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm) override;
  virtual void
  MakeNewLevelFromCoarse(int level, amrex::Real time, const amrex::BoxArray &ba,
                         const amrex::DistributionMapping &dm) override;
  virtual void RemakeLevel(int level, amrex::Real time,
                           const amrex::BoxArray &ba,
                           const amrex::DistributionMapping &dm) override;
  virtual void ClearLevel(int level) override;
};

#ifdef HAVE_CCTK_REAL2
// Storage for CCTK_REAL2 (binary16, `_Float16`) grid function groups. There
// is no `amrex::hMultiFab` typedef upstream (unlike `amrex::fMultiFab` for
// float), so we define our own here.
using hMultiFab = amrex::FabArray<amrex::BaseFab<CCTK_REAL2> >;
#endif

// Storage for a grid function group's data, at the precision (CCTK_REAL,
// CCTK_REAL4, or CCTK_REAL2) determined by the group's `vartype`
using AnyMultiFab = std::variant<amrex::MultiFab, amrex::fMultiFab
#ifdef HAVE_CCTK_REAL2
                                 ,
                                 hMultiFab
#endif
                                 >;

// Storage precision of a Cactus real vartype: REAL and REAL8 are double
// (CCTK_REAL_PRECISION==8 is asserted above), REAL4 is float, REAL2 is
// _Float16 (if HAVE_CCTK_REAL2). CCTK_REAL16 is not supported.
inline bool vartype_is_real4(const int vartype) {
  return vartype == CCTK_VARIABLE_REAL4;
}
inline bool vartype_is_real8(const int vartype) {
  return vartype == CCTK_VARIABLE_REAL || vartype == CCTK_VARIABLE_REAL8;
}
inline bool vartype_is_real2(const int vartype) {
#ifdef HAVE_CCTK_REAL2
  return vartype == CCTK_VARIABLE_REAL2;
#else
  return false;
#endif
}
inline bool vartype_is_supported_real(const int vartype) {
  return vartype_is_real4(vartype) || vartype_is_real8(vartype) ||
         vartype_is_real2(vartype);
}

// Compute type for a storage type T: for CCTK_REAL2 (_Float16) storage,
// which has too little dynamic range and precision to safely accumulate
// several values in and whose arithmetic is ambiguous when mixed with plain
// int/double literals on some toolchains (nvcc/__half), widen to `float`;
// for every other T, the compute type is T itself. This is the
// "storage precision != compute precision" policy applied wherever several
// values of a group's storage type are combined (restriction, prolongation)
// -- widen once on load, accumulate/interpolate in the compute type, narrow
// once on store. `prolongate_3d_rf2_impl.hxx`'s `prolongate_compute_t<T>` is
// an alias of this (kept as a separate name there for readability at its
// many call sites); `restrict_edges.hxx` uses this name directly.
#ifdef HAVE_CCTK_REAL2
template <typename T>
using compute_t = std::conditional_t<std::is_same_v<T, CCTK_REAL2>, float, T>;
#else
template <typename T> using compute_t = T;
#endif

inline bool is_real4(const AnyMultiFab &mfab) {
  return std::holds_alternative<amrex::fMultiFab>(mfab);
}
#ifdef HAVE_CCTK_REAL2
inline bool is_real2(const AnyMultiFab &mfab) {
  return std::holds_alternative<hMultiFab>(mfab);
}
#endif

inline amrex::MultiFab &as_mfab_real(AnyMultiFab &mfab) {
  return std::get<amrex::MultiFab>(mfab);
}
inline const amrex::MultiFab &as_mfab_real(const AnyMultiFab &mfab) {
  return std::get<amrex::MultiFab>(mfab);
}

template <typename... Args>
std::unique_ptr<AnyMultiFab> make_any_mfab(const int vartype, Args &&...args) {
  assert(vartype_is_supported_real(vartype));
#ifdef HAVE_CCTK_REAL2
  if (vartype_is_real2(vartype))
    return std::make_unique<AnyMultiFab>(std::in_place_type<hMultiFab>,
                                         std::forward<Args>(args)...);
#endif
  if (vartype_is_real4(vartype))
    return std::make_unique<AnyMultiFab>(
        std::in_place_type<amrex::fMultiFab>, std::forward<Args>(args)...);
  return std::make_unique<AnyMultiFab>(std::in_place_type<amrex::MultiFab>,
                                       std::forward<Args>(args)...);
}

// Cactus grid hierarchy extension
struct GHExt {

  GHExt() = default;
  GHExt(const GHExt &) = delete;
  GHExt(GHExt &&) = delete;
  GHExt &operator=(const GHExt &) = delete;
  GHExt &operator=(GHExt &&) = delete;

  struct cctkGHptr {
    cGH *cctkGH;
    cctkGHptr(const cctkGHptr &) = delete;
    cctkGHptr(cctkGHptr &&ptr) : cctkGH(ptr.cctkGH) { ptr.cctkGH = nullptr; }
    cctkGHptr &operator=(const cctkGHptr &) = delete;
    cctkGHptr &operator=(cctkGHptr &&ptr);
    cctkGHptr() : cctkGH(nullptr) {}
    cctkGHptr(cGH *&&cctkGH) : cctkGH(cctkGH) {}
    cctkGHptr &operator=(cGH *&&cctkGH);
    ~cctkGHptr();
    operator bool() const { return bool(cctkGH); }
    cGH *get() const { return cctkGH; }
  };

  cctkGHptr global_cctkGH;
  std::vector<cctkGHptr> level_cctkGHs; // [reflevel]

  struct CommonGroupData {
    std::string groupname;
    int groupindex;
    int firstvarindex;
    int numvars;

    bool do_checkpoint; // whether to checkpoint
    bool do_restrict;   // whether to restrict

    std::vector<std::vector<why_valid_t> > valid; // [time level][var index]

    // TODO: add poison_invalid and check_valid functions

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const CommonGroupData &commongroupdata);
  };

  struct GlobalData {
    // all data that exists on all levels

    class AnyTypeVector {

    public:
      // access to a single element of a AnyTypeVector
      class AnyTypeScalarRef {
      public:
        AnyTypeScalarRef() = delete;
        AnyTypeScalarRef(const AnyTypeVector &vect_, size_t idx_)
            : _vect(vect_), _idx(idx_) {}

      private:
        const AnyTypeVector &_vect;
        const size_t _idx;

        friend YAML::Emitter &
        operator<<(YAML::Emitter &yaml,
                   const AnyTypeScalarRef &anytypescalarref);
        friend std::ostream &operator<<(std::ostream &os,
                                        const AnyTypeScalarRef &scalar);
      };

      AnyTypeVector() : _type(-1), _typesize(-1), _count(0), _data(nullptr) {};
      AnyTypeVector(int type_, size_t count_)
          : _type(-1), _typesize(-1), _count(0), _data(nullptr) {
        alloc(type_, count_);
        assert(_type == type_);
        assert(_typesize != -1);
        assert(_count == count_);
        assert(_data != nullptr);
      };
      // Noncopyable for now
      AnyTypeVector(const AnyTypeVector &) = delete;
      AnyTypeVector &operator=(const AnyTypeVector &) = delete;
      AnyTypeVector &operator=(AnyTypeVector &&other) {
        swap(other);
        return *this;
      }
      AnyTypeVector(AnyTypeVector &&other)
          : _type(other._type), _typesize(other._typesize),
            _count(other._count), _data(other._data) {
        other._type = -1;
        other._typesize = -1;
        other._count = 0;
        other._data = nullptr;
      }
      void swap(AnyTypeVector &other) {
        std::swap(this->_type, other._type);
        std::swap(this->_typesize, other._typesize);
        std::swap(this->_count, other._count);
        std::swap(this->_data, other._data);
      }

      ~AnyTypeVector() {
        if (_data != nullptr) {
          assert(_type != -1);
          assert(_typesize != -1);
          amrex::The_Arena()->free(_data);
          _type = -1;
          _typesize = -1;
          _count = 0;
          _data = nullptr;
        }
        assert(_type == -1);
        assert(_typesize == -1);
        assert(_count == 0);
        assert(_data == nullptr);
      };

      void alloc(int type_, size_t count_) {
        assert(type_ == CCTK_VARIABLE_INT ||
               type_ == CCTK_VARIABLE_COMPLEX ||
               vartype_is_supported_real(type_));

        assert(_type == -1);
        assert(_typesize == -1);
        assert(_count == 0);
        assert(_data == nullptr);

        _type = type_;
        _typesize = CCTK_VarTypeSize(_type);
        assert(_typesize > 0);
        _count = count_;
        _data = amrex::The_Arena()->alloc(_typesize * _count);
      }

      void free() {
        assert(_type != -1);
        assert(_typesize != -1);
        assert(_data != nullptr);
        amrex::The_Arena()->free(_data);
        _type = -1;
        _typesize = -1;
        _count = 0;
        _data = nullptr;
      }

      int type() const { return _type; };
      int typesize() const { return _typesize; };

      const void *data_at(size_t i) const {
#ifdef CCTK_DEBUG
        if (i >= _count) {
          CCTK_VERROR("invalid index %zd exceeds %zd", i, _count);
        }
#endif
        assert(i < _count);
        return (char *)_data + i * _typesize;
      };

      void *data_at(size_t i) {
#ifdef CCTK_DEBUG
        if (i >= _count) {
          CCTK_VERROR("invalid index %zu exceeds %zu", i, _count);
        }
#endif
        assert(i < _count);
        return (char *)_data + i * _typesize;
      };

      AnyTypeScalarRef operator[](size_t idx) const {
        return AnyTypeScalarRef(*this, idx);
      }

      size_t size() const { return _count; };

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const AnyTypeVector &anytypevector);

    private:
      int _type, _typesize;
      size_t _count;
      void *_data;
    };

    // For subcycling in time, there really should be one copy of each
    // integrated grid scalar per level. We don't do that yet; instead,
    // we assume that grid scalars only hold "analysis" data.

    struct ArrayGroupData : public CommonGroupData {
      std::vector<AnyTypeVector>
          data; // [time level][var index + grid point index]
      int array_size;
      int dimension;
      int activetimelevels;
      int lsh[dim];
      int ash[dim];
      int gsh[dim];
      int lbnd[dim];
      int ubnd[dim];
      int bbox[2 * dim];
      int nghostzones[dim];

      ArrayGroupData() {
        array_size = -1;
        dimension = -1;
        activetimelevels = -1;
        for (int d = 0; d < dim; d++) {
          lsh[d] = -1;
          ash[d] = -1;
          gsh[d] = -1;
          lbnd[d] = -1;
          ubnd[d] = -1;
          bbox[2 * d] = bbox[2 * d + 1] = -1;
          nghostzones[d] = -1;
        }
      }

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const ArrayGroupData &arraygroupdata);
    };
    // TODO: right now this is sized for the total number of groups
    std::vector<std::unique_ptr<ArrayGroupData> >
        arraygroupdata; // [group index]

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const GlobalData &globaldata);
  };
  GlobalData globaldata;

  struct PatchData {
    PatchData() = delete;
    PatchData(const PatchData &) = delete;
    PatchData &operator=(const PatchData &) = delete;
    PatchData(PatchData &&) = default;
    PatchData &operator=(PatchData &&) = default;

    PatchData(int patch);

    int patch;

    bool is_cartesian;

    std::array<std::array<symmetry_t, dim>, 2> symmetries;

    // AMReX grid structure
    // TODO: convert this from unique_ptr to optional
    std::unique_ptr<CactusAmrCore> amrcore;

    struct LevelData {
      LevelData() = delete;
      LevelData(const LevelData &) = delete;
      LevelData &operator=(const LevelData &) = delete;
      LevelData(LevelData &&) = default;
      LevelData &operator=(LevelData &&) = default;

      LevelData(const int patch, const int level, const amrex::BoxArray &ba,
                const amrex::DistributionMapping &dm,
                const std::function<std::string()> &why);

      int patch, level;

      // This level uses subcycling with respect to the next coarser
      // level. (Ignored for the coarsest level.)
      bool is_subcycling_level;

      // Iteration and time at which this cycle level is valid
      rat64 iteration, delta_iteration;

      // Fabamrex::ArrayBase object holding a cell-centred BoxArray for
      // iterating over grid functions. This stores the grid structure
      // and its distribution over all processes, but holds no data.
      std::unique_ptr<amrex::FabArrayBase> fab;

      cctkGHptr patch_cctkGH;
      std::vector<cctkGHptr> local_cctkGHs; // [component]

      cGH *get_patch_cctkGH() const { return patch_cctkGH.get(); }
      cGH *get_local_cctkGH(const int component) const {
        return local_cctkGHs.at(component).get();
      }

      struct GroupData : public CommonGroupData {
        GroupData() = delete;
        GroupData(const GroupData &) = delete;
        GroupData &operator=(const GroupData &) = delete;
        GroupData(GroupData &&) = delete;
        GroupData &operator=(GroupData &&) = delete;

        GroupData(int patch, int level, int gi, const amrex::BoxArray &ba,
                  const amrex::DistributionMapping &dm,
                  const std::function<std::string()> &why);

        int patch, level;

        int vartype;

        std::array<int, dim> indextype;
        std::array<int, dim> nghostzones;

        // Prolongation operator for this group, selected (once, at
        // construction, from `vartype`) from the precision-appropriate
        // static instance table in prolongate_3d_rf2_impl_*.cxx via
        // get_interpolator_t<T>() (driver.cxx). Exactly one of
        // `interpolator_real8`/`interpolator_real4`/`interpolator_real2` is
        // non-null for any given group -- REAL/REAL8 (amrex::MultiFab-backed)
        // groups get `interpolator_real8`, CCTK_REAL4
        // (amrex::fMultiFab-backed) ones get `interpolator_real4`,
        // CCTK_REAL2 (hMultiFab-backed) ones get `interpolator_real2` --
        // mirroring AnyMultiFab's own precision split. fillpatch.cxx's
        // templated FillPatch_*<MF> functions and their callers
        // (schedule.cxx, driver.cxx's MakeNewLevelFromCoarse/RemakeLevel)
        // select whichever of these matches the group's AnyMultiFab
        // alternative.
        //
        // interpolator_real2 is populated and consumed the same way as the
        // other two precisions (the prolongate_3d_rf2_impl_*.cxx tables have
        // a T=CCTK_REAL2 axis, see prolongate_3d_rf2.hxx's
        // CARPETX_DECLARE_PROLONGATE_TABLE): fillpatch.cxx's templated
        // FillPatch_*<MF> functions are explicitly instantiated for
        // MF=hMultiFab, and driver.cxx's MakeNewLevelFromCoarse/RemakeLevel
        // and schedule.cxx's SyncGroupsByDirI route REAL2 groups through
        // them.
        InterpolaterT<CCTK_REAL> *interpolator_real8 = nullptr;
        InterpolaterT<CCTK_REAL4> *interpolator_real4 = nullptr;
#ifdef HAVE_CCTK_REAL2
        InterpolaterT<CCTK_REAL2> *interpolator_real2 = nullptr;
#endif

        std::array<std::array<boundary_t, dim>, 2> boundaries;
        bool all_faces_have_symmetries_or_boundaries() const;
        std::vector<std::array<int, dim> > parities;
        std::vector<CCTK_REAL> dirichlet_values;
        std::vector<CCTK_REAL> robin_values;
        amrex::Vector<amrex::BCRec> bcrecs;

        // Apply outer (physical) boundary conditions to a MultiFab or
        // fMultiFab, whichever matches this group's precision (i.e. the
        // AnyMultiFab alternative held in `mfab`/`tmp_mfabs` below).
        // Explicitly instantiated for amrex::MultiFab and amrex::fMultiFab
        // in driver.cxx.
        template <typename MF> void apply_boundary_conditions(MF &mfab) const;

        // each amrex::MultiFab (or amrex::fMultiFab, for CCTK_REAL4
        // groups) has numvars components
        std::vector<std::unique_ptr<AnyMultiFab> > mfab; // [time level]

        // flux register between this and the next coarser level
        // (REAL groups only; CCTK_REAL4 groups cannot have flux registers)
        std::unique_ptr<amrex::FluxRegister> freg;
        // associated flux group indices
        std::array<int, dim> fluxes; // [dir]

        // CarpetX can allocate and free (temporary) multifabs that
        // are associated with a Cactus grid function group. These
        // multifabs remain allocated when they are freed, which makes
        // it efficient when they are re-allocated later. However,
        // they are freed when the current level changes during
        // regridding (and the shape of the multifab presumably
        // changes). This is used e.g. by ODESolvers for its
        // temporaries.
      private:
        mutable std::vector<std::unique_ptr<AnyMultiFab> > tmp_mfabs;
        mutable std::size_t next_tmp_mfab;

      public:
        void init_tmp_mfabs() const;
        AnyMultiFab *alloc_tmp_mfab() const;
        void free_tmp_mfabs() const;

        friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                         const GroupData &groupdata);
      };
      // TODO: right now this is sized for the total number of groups
      std::vector<std::unique_ptr<GroupData> > groupdata; // [group index]

      friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                       const LevelData &leveldata);
    };
    std::vector<LevelData> leveldata; // [reflevel]

    friend YAML::Emitter &operator<<(YAML::Emitter &yaml,
                                     const PatchData &patchdata);
  };
  std::vector<PatchData> patchdata; // [patch]

  int num_patches() const { return patchdata.size(); }
  int num_levels(const int patch) const {
    return patchdata.at(patch).leveldata.size();
  }
  int num_levels() const {
    int nlevels = 0;
    using std::max;
    for (const auto &pd : patchdata)
      nlevels = max(nlevels, int(pd.leveldata.size()));
    return nlevels;
  }

  cGH *get_global_cctkGH() const { return global_cctkGH.get(); }
  cGH *get_level_cctkGH(const int level) const {
    return level_cctkGHs.at(level).get();
  }
  cGH *get_patch_cctkGH(const int level, const int patch) const {
    return patchdata.at(patch).leveldata.at(level).patch_cctkGH.get();
  }
  cGH *get_local_cctkGH(const int level, const int patch,
                        const int component) const {
    return patchdata.at(patch)
        .leveldata.at(level)
        .local_cctkGHs.at(component)
        .get();
  }

  friend YAML::Emitter &operator<<(YAML::Emitter &yaml, const GHExt &ghext);
  friend std::ostream &operator<<(std::ostream &os, const GHExt &ghext);
};

extern std::unique_ptr<GHExt> ghext;

// GroupData::apply_boundary_conditions<MF>() is explicitly instantiated for
// both storage precisions in driver.cxx.
extern template void
GHExt::PatchData::LevelData::GroupData::apply_boundary_conditions<
    amrex::MultiFab>(amrex::MultiFab &mfab) const;
extern template void
GHExt::PatchData::LevelData::GroupData::apply_boundary_conditions<
    amrex::fMultiFab>(amrex::fMultiFab &mfab) const;
#ifdef HAVE_CCTK_REAL2
extern template void
GHExt::PatchData::LevelData::GroupData::apply_boundary_conditions<hMultiFab>(
    hMultiFab &mfab) const;
#endif

// Monotonically increasing counter. Incremented whenever the AMR grid
// hierarchy is invalidated (regridding, recovery). Starts at 0.
extern std::atomic<CCTK_INT> carpetx_epoch;

extern "C" CCTK_INT CarpetX_GetEpoch(void);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_DRIVER_HXX
