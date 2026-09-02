#include "io_openpmd.hxx"

#include "driver.hxx"
#include "io_real2.hxx"
#include "timer.hxx"

#include <div.hxx>
#include <vect.hxx>

#include <CactusBase/IOUtil/src/ioutil_CheckpointRecovery.h>
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <util_Network.h>

#ifdef HAVE_CAPABILITY_openPMD_api

#include <openPMD/openPMD.hpp>

#ifdef HAVE_CAPABILITY_ADIOS2
#include <adios2.h>
#endif

#if defined _OPENMP
#include <omp.h>
#elif defined __HIPCC__
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_in_parallel() { return 0; }
#endif

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ios>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if !openPMD_HAVE_MPI
#error                                                                         \
    "CarpetX requires openPMD_api with MPI support. Please use -DopenPMD_USE_MPI=ON when building openPMD."
#endif

namespace CarpetX {

constexpr bool io_verbose = true;

openPMD::Format get_format() {
  DECLARE_CCTK_PARAMETERS;
  if (CCTK_EQUALS(openpmd_format, "HDF5"))
    return openPMD::Format::HDF5;
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS1"))
    return openPMD::Format::ADIOS1;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_auto"))
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
    return openPMD::Format::ADIOS2_BP5;
#else
    return openPMD::Format::ADIOS2;
#endif
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP"))
    return openPMD::Format::ADIOS2_BP;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP4"))
    return openPMD::Format::ADIOS2_BP4;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_BP5"))
    return openPMD::Format::ADIOS2_BP5;
#else
  if (CCTK_EQUALS(openpmd_format, "ADIOS2"))
    return openPMD::Format::ADIOS2;
#endif
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SST"))
    return openPMD::Format::ADIOS2_SST;
  if (CCTK_EQUALS(openpmd_format, "ADIOS2_SSC"))
    return openPMD::Format::ADIOS2_SSC;
  if (CCTK_EQUALS(openpmd_format, "JSON"))
    return openPMD::Format::JSON;
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
  if (CCTK_EQUALS(openpmd_format, "TOML"))
    return openPMD::Format::TOML;
#endif
#if OPENPMDAPI_VERSION_GE(0, 16, 0)
  if (CCTK_EQUALS(openpmd_format, "GENERIC"))
    return openPMD::Format::GENERIC;
#endif
  CCTK_VERROR("The openPMD format \"%s\" is not supported in version %d.%d.%d "
              "of the openPMD_api library",
              openpmd_format, OPENPMDAPI_VERSION_MAJOR,
              OPENPMDAPI_VERSION_MINOR, OPENPMDAPI_VERSION_PATCH);
}

// - fileBased: One file per iteration. Needs templated file name to encode
//   iteration number.
// - groupBased: Multiple iterations per file
// - variableBased: Multiple iterations stored per variable. Needs special
//    support in the backend.
constexpr openPMD::IterationEncoding iterationEncoding =
    openPMD::IterationEncoding::fileBased;

// TODO: Set number of threads?
#ifdef ADIOS2_HAVE_BLOSC2
const std::string options = R"EOS(
  {
    "adios2": {
      "dataset": {
        "operators": [
          {
            "type": "blosc",
            "parameters": {
              "clevel": "9",
              "doshuffle": "BLOSC_SHUFFLE"
            }
          }
        ]
      }
    }
  }
)EOS";
#else
const std::string options = R"EOS(
  {
    "adios2": {
      "dataset": {
        "operators": [
        ]
      }
    }
  }
)EOS";
#endif

constexpr bool input_ghosts = false;
constexpr bool output_ghosts = false;

////////////////////////////////////////////////////////////////////////////////

struct Const {
  // From: CODATA Internationally recommended 2018 values of the
  // Fundamental Physical Constants
  static constexpr CCTK_REAL c = 299792458;         // m s⁻¹
  static constexpr CCTK_REAL G = 6.67430e-11;       // m³ kg⁻¹ s⁻²
  static constexpr CCTK_REAL M_solar = 1.98847e+30; // kg
};

struct Unit {
  // We use c = G = 1, and M_solar as mass unit.
  static constexpr CCTK_REAL velocity = Const::c;   // m s⁻¹
  static constexpr CCTK_REAL mass = Const::M_solar; // kg
  static constexpr CCTK_REAL length = Const::G * mass / pow2(Const::c); // m
  static constexpr CCTK_REAL time = length / velocity;                  // s
};

////////////////////////////////////////////////////////////////////////////////

namespace {

// TODO: Get this from `valid.cxx`
inline CCTK_REAL get_poison() {
#if defined CCTK_REAL_PRECISION_4
  constexpr std::uint32_t ipoison = 0xffc00000UL + 0xdead;
#elif defined CCTK_REAL_PRECISION_8
  constexpr std::uint64_t ipoison = 0xfff8000000000000ULL + 0xdeadbeef;
#endif
  static_assert(sizeof ipoison == sizeof(CCTK_REAL));
  CCTK_REAL poison;
  std::memcpy(&poison, &ipoison, sizeof poison);
  return poison;
}

template <typename T, std::size_t N>
constexpr std::array<T, N> reversed(const std::array<T, N> &arr) {
  std::array<T, N> res;
  for (std::size_t n = 0; n < N; ++n)
    res[n] = arr[N - 1 - n];
  return res;
}

template <typename T, int D>
constexpr Arith::vect<T, D> reversed(const Arith::vect<T, D> &arr) {
  Arith::vect<T, D> res;
  for (std::size_t d = 0; d < D; ++d)
    res[d] = arr[D - 1 - d];
  return res;
}

template <typename T> std::vector<T> reversed(const std::vector<T> &vec) {
  std::vector<T> res(vec.size());
  std::reverse_copy(vec.begin(), vec.end(), res.begin());
  return res;
}
template <typename T> std::vector<T> reversed(std::vector<T> &&vec) {
  std::vector<T> res(vec);
  std::reverse(res.begin(), res.end());
  return res;
}

template <typename T = std::uint64_t, typename C>
std::vector<T> to_vector(const C &source) {
  std::vector<T> res(source.size());
  for (std::size_t n = 0; n < res.size(); ++n)
    res[n] = source[n];
  return res;
}

template <typename T> std::string vector_string(const std::vector<T> &vec) {
  std::ostringstream buf;
  buf << "[";
  for (std::size_t n = 0; n < vec.size(); ++n) {
    if (n != 0)
      buf << ",";
    buf << vec[n];
  }
  buf << "]";
  return buf.str();
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

std::string Geometry_string(const openPMD::Mesh::Geometry &geometry) {
  std::ostringstream buf;
  buf << geometry;
  return buf.str();
}

std::string UnitDimension_string(const std::array<double, 7> &unitDimension) {
  std::ostringstream buf;
  buf << "L:" << unitDimension[uint8_t(openPMD::UnitDimension::L)] << " "
      << "M:" << unitDimension[uint8_t(openPMD::UnitDimension::M)] << " "
      << "T:" << unitDimension[uint8_t(openPMD::UnitDimension::T)] << " "
      << "I:" << unitDimension[uint8_t(openPMD::UnitDimension::I)] << " "
      << "θ:" << unitDimension[uint8_t(openPMD::UnitDimension::theta)] << " "
      << "N:" << unitDimension[uint8_t(openPMD::UnitDimension::N)] << " "
      << "J:" << unitDimension[uint8_t(openPMD::UnitDimension::J)];
  return buf.str();
}

////////////////////////////////////////////////////////////////////////////////

struct carpetx_openpmd_t {
  carpetx_openpmd_t() = default;

  carpetx_openpmd_t(const carpetx_openpmd_t &) = delete;
  carpetx_openpmd_t(carpetx_openpmd_t &&) = default;
  carpetx_openpmd_t &operator=(const carpetx_openpmd_t &) = delete;
  carpetx_openpmd_t &operator=(carpetx_openpmd_t &&) = default;

  static std::optional<carpetx_openpmd_t> self;

  ////////////////////////////////////////////////////////////////////////////////

  template <typename T, std::size_t D> struct box_t {
    Arith::vect<T, D> lo, hi;
    constexpr friend bool operator==(const box_t &x, const box_t &y) {
      if (x.empty() && y.empty())
        return true;
      if (x.empty() || y.empty())
        return false;
      return all(x.lo == y.lo) && all(x.hi == y.hi);
    }
    constexpr friend bool operator!=(const box_t &x, const box_t &y) {
      return !(x == y);
    }
    constexpr bool empty() const { return any(hi < lo); }
    constexpr Arith::vect<T, D> shape() const {
      Arith::vect<T, D> sh;
      for (std::size_t d = 0; d < D; ++d)
        sh[d] = hi[d] < lo[d] ? T{0} : hi[d] - lo[d];
      return sh;
    }
    constexpr T size() const {
      const auto sh = shape();
      T sz{1};
      for (std::size_t d = 0; d < D; ++d)
        sz *= sh[d];
      return sz;
    }
    constexpr Arith::vect<T, D> stride() const {
      const Arith::vect<T, D> sh = shape();
      Arith::vect<T, D> str;
      T np{1};
      for (std::size_t d = 0; d < D; ++d)
        str[d] = (np *= sh[d]);
      return str;
    }
    constexpr T linear(const Arith::vect<T, D> &index) const {
      const Arith::vect<T, D> str = stride();
      T lin{0};
      if (D > 0) {
        lin = index[0];
        for (std::size_t d = 1; d < D; ++d)
          lin += index[d] * str[d - 1];
      }
      return lin;
    }
    friend std::ostream &operator<<(std::ostream &os, const box_t<T, D> &box) {
      return os << "box_t{lo:" << box.lo << ",hi:" << box.hi << "}";
    }
  };

  template <typename T, typename I, std::size_t D> struct level_t {
    box_t<T, D> rdomain;
    Arith::vect<bool, D> is_cell_centred;
    box_t<I, D> idomain;
    constexpr Arith::vect<T, D> rcoord(const Arith::vect<I, D> &icoord) const {
      Arith::vect<T, D> r;
      for (std::size_t d = 0; d < D; ++d) {
        const T rlo = rdomain.lo[d];
        const T rhi = rdomain.hi[d];
        const I ilo2 = 2 * idomain.lo[d] - is_cell_centred;
        const I ihi2 = 2 * idomain.hi[d] + is_cell_centred;
        const I i2 = 2 * icoord[d];
        // The expression below has been carefully constructed to
        // avoid round-off errors at i=ilo and i=ihi. Do not rearrange
        // the terms without preserving this property.
        r[d] = (T(ihi2 - i2) / T(ihi2 - ilo2)) * rlo[d] +
               (T(i2 - ilo2) / T(ihi2 - ilo2)) * rhi[d];
      }
      return r;
    }
    std::vector<box_t<I, D> > grids;
    Arith::vect<std::vector<I>, 2> offsets_sizes() const {
      std::vector<I> offsets(grids.size() + 1), sizes(grids.size());
      I offset{0};
      for (std::size_t n = 0; n < grids.size(); ++n) {
        const T size = grids[n].size();
        offsets[n] = offset;
        sizes[n] = size;
        offset += size;
      }
      offsets[grids.size()] = offset;
      return {offsets, sizes};
    }
  };

  template <typename T, typename I, std::size_t D> struct grid_structure_t {
    box_t<T, D> rdomain;
    std::vector<level_t<T, I, D> > levels;
  };

  ////////////////////////////////////////////////////////////////////////////////

  // Allowed characters are only [A-Za-z_]
  static std::string make_meshname(const int gi, const int patch,
                                   const int level) {
    std::string groupname = CCTK_FullGroupName(gi);
    groupname = std::regex_replace(groupname, std::regex("::"), "_");
    for (auto &ch : groupname)
      ch = std::tolower(ch);
    std::ostringstream buf;
    buf << groupname;
    if (patch != -1)
      buf << "_patch" << std::setw(2) << std::setfill('0') << patch;
    if (level != -1)
      // The suffix should be `_lvl<N>`. No `std::setfill`?
      buf << "_lev" << std::setw(2) << std::setfill('0') << level;
    return buf.str();
  }

#if 0
  static std_tuple<int, int> interpret_meshname(const std::string &meshname) {
    std::smatch match;
    const bool matched =
        std::regex_match(meshname, match, std::regex("(\\w+)_lev0*(\\d+)"));
    if (!matched)
      CCTK_VERROR("Cannot parse mesh name %s", meshname.c_str());
    const std::string groupname = match[1].str();
    const int gi = CCTK_GroupIndex(groupname.c_str());
    if (gi < 0)
      CCTK_VERROR("Unknown group name %s", groupname.c_str());
    const std::string levelstr = match[2].str();
    const int level = std::stoi(levelstr);
    return {gi, level};
  }
#endif

  // Allowed characters are only [A-Za-z_]
  static std::string make_componentname(const int gi, const int vi) {
    const int v0 = CCTK_FirstVarIndexI(gi);
    std::string varname = CCTK_FullVarName(v0 + vi);
    varname = std::regex_replace(varname, std::regex("::"), "_");
    for (auto &ch : varname)
      ch = std::tolower(ch);
    return varname;
  }

  ////////////////////////////////////////////////////////////////////////////////

  std::optional<std::string> filename;
  std::optional<openPMD::Series> series;
  // std::optional<openPMD::ReadIterations> read_iters;
  std::optional<openPMD::Iteration> read_iter;
  std::optional<openPMD::WriteIterations> write_iters;

  int InputOpenPMDParameters(const std::string &input_dir,
                             const std::string &input_file);
  void InputOpenPMDGridStructure(cGH *cctkGH, const std::string &input_dir,
                                 const std::string &input_file,
                                 int input_iteration);
  void InputOpenPMD(const cGH *const cctkGH,
                    const std::vector<bool> &input_group,
                    const std::string &input_dir,
                    const std::string &input_file, io_mode mode);

  void OutputOpenPMD(const cGH *const cctkGH,
                     const std::vector<bool> &output_group,
                     const std::string &output_dir,
                     const std::string &output_file, io_mode mode);
};

////////////////////////////////////////////////////////////////////////////////

std::optional<carpetx_openpmd_t> carpetx_openpmd_t::self;

int InputOpenPMDParameters(const std::string &input_dir,
                           const std::string &input_file) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  return carpetx_openpmd_t::self->InputOpenPMDParameters(input_dir, input_file);
}
void InputOpenPMDGridStructure(cGH *cctkGH, const std::string &input_dir,
                               const std::string &input_file,
                               int input_iteration) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->InputOpenPMDGridStructure(
      cctkGH, input_dir, input_file, input_iteration);
}
void InputOpenPMD(const cGH *cctkGH, const std::vector<bool> &input_group,
                  const std::string &input_dir, const std::string &input_file,
                  const io_mode mode) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->InputOpenPMD(cctkGH, input_group, input_dir,
                                        input_file, mode);
}

void OutputOpenPMD(const cGH *const cctkGH,
                   const std::vector<bool> &output_group,
                   const std::string &output_dir,
                   const std::string &output_file, const io_mode mode) {
  if (!carpetx_openpmd_t::self)
    carpetx_openpmd_t::self = std::make_optional<carpetx_openpmd_t>();
  carpetx_openpmd_t::self->OutputOpenPMD(cctkGH, output_group, output_dir,
                                         output_file, mode);
}

void ShutdownOpenPMD() { carpetx_openpmd_t::self.reset(); }

////////////////////////////////////////////////////////////////////////////////

int carpetx_openpmd_t::InputOpenPMDParameters(const std::string &input_dir,
                                              const std::string &input_file) {
  DECLARE_CCTK_PARAMETERS;

  assert(!input_dir.empty());
  assert(!input_file.empty());

  // Set up timers
  static Timer timer("InputOpenPMDParameters");
  Interval interval(timer);

  if (io_verbose)
    CCTK_VINFO("InputOpenPMDParameters...");

  const openPMD::Format format = get_format();

  int input_iteration = -1;

  assert(!series);
  if (!series) {
    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << input_dir << "/" << input_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << input_dir << "/" << input_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    try {
      series = std::make_optional<openPMD::Series>(
          *filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD, options);
    } catch (const openPMD::no_such_file_error &) {
      // Did not find a checkpoint file
      if (io_verbose) {
        CCTK_VINFO("Not recovering parameters:");
        CCTK_VINFO("  Could not find an openPMD checkpoint file \"%s\"",
                   filename->c_str());
      }
      return -1; // no iteration found
    }

    // Find largest iteration
    for (auto iter = series->iterations.begin();
         iter != series->iterations.end(); ++iter)
      input_iteration = iter->first;

    if (input_iteration < 0) {
      // Did not find a checkpoint file
      if (io_verbose) {
        CCTK_VINFO("Not recovering parameters:");
        CCTK_VINFO("  Could not find an openPMD checkpoint file \"%s\"",
                   filename->c_str());
      }
      return -1; // no iteration found
    }

    read_iter = std::make_optional<openPMD::Iteration>(
        series->iterations[input_iteration]);
  }
  assert(filename);
  assert(series);
  // assert(read_iters);
  assert(read_iter);

  CCTK_VINFO("Recovering parameters from checkpoint file \"%s\" iteration %d",
             filename->c_str(), input_iteration);

  // Read metadata
  {
    const bool has_parameters = read_iter->containsAttribute("AllParameters");
    assert(has_parameters);
    const openPMD::Attribute parameters_attr =
        read_iter->getAttribute("AllParameters");
    assert(parameters_attr.dtype == openPMD::Datatype::STRING);
    const std::string parameters = parameters_attr.get<std::string>();
    IOUtil_SetAllParameters(parameters.data());
  }

  return input_iteration;
}

void carpetx_openpmd_t::InputOpenPMDGridStructure(cGH *cctkGH,
                                                  const std::string &input_dir,
                                                  const std::string &input_file,
                                                  const int input_iteration) {
  DECLARE_CCTK_PARAMETERS;

  assert(!input_dir.empty());
  assert(!input_file.empty());
  assert(input_iteration >= 0);

  // Set up timers
  static Timer timer("InputOpenPMDGridStructure");
  Interval interval(timer);

  if (io_verbose)
    CCTK_VINFO("InputOpenPMDGridStructure...");

  assert(filename);
  assert(series);
  // assert(read_iters);
  assert(read_iter);

  cctkGH->cctk_iteration = input_iteration;
  cctkGH->cctk_time = read_iter->time<double>();

  // TODO: Check whether attribute exists and has correct type
  const int ndims = read_iter->getAttribute("numDims").get<std::int64_t>();
  assert(ndims >= 0);

  const int npatches =
      read_iter->getAttribute("numPatches").get<std::int64_t>();
  if (npatches != ghext->num_patches())
    CCTK_VERROR(
        "Wrong number of patches: Expected %d, found %d in the checkpoint file",
        ghext->num_patches(), npatches);

  std::vector<std::string> patch_suffixes;
  if (npatches == 1) {
    assert(read_iter->getAttribute("patchSuffixes").dtype ==
           openPMD::Datatype::STRING);
    const std::string patch_suffix =
        read_iter->getAttribute("patchSuffixes").get<std::string>();
    patch_suffixes.resize(1);
    patch_suffixes.at(0) = patch_suffix;
  } else {
    assert(read_iter->getAttribute("patchSuffixes").dtype ==
           openPMD::Datatype::VEC_STRING);
    patch_suffixes = read_iter->getAttribute("patchSuffixes")
                         .get<std::vector<std::string> >();
  }

  for (auto &patchdata : ghext->patchdata) {
    const int patch = patchdata.patch;
    const int nlevels =
        read_iter->getAttribute("numLevels" + patch_suffixes.at(patch))
            .get<std::int64_t>();
    assert(nlevels >= 0);
    std::vector<std::string> level_suffixes;
    if (nlevels == 1) {
      assert(read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
                 .dtype == openPMD::Datatype::STRING);
      const std::string level_suffix =
          read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
              .get<std::string>();
      level_suffixes.resize(1);
      level_suffixes.at(0) = level_suffix;
    } else {
      assert(read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
                 .dtype == openPMD::Datatype::VEC_STRING);
      level_suffixes =
          read_iter->getAttribute("levelSuffixes" + patch_suffixes.at(patch))
              .get<std::vector<std::string> >();
    }
    assert(int(level_suffixes.size()) == nlevels);

    assert(ndims == 3);
    assert(nlevels > 0);
    patchdata.amrcore->SetFinestLevel(nlevels - 1);

    for (int level = 0; level < nlevels; ++level) {
      const std::vector<std::int64_t> chunk_infos =
          read_iter->getAttribute("chunkInfo" + level_suffixes.at(level))
              .get<std::vector<std::int64_t> >();
      assert(chunk_infos.size() % (2 * ndims) == 0);
      const int nfabs = chunk_infos.size() / (2 * ndims);
      amrex::Vector<amrex::Box> levboxes(nfabs);
      for (int component = 0; component < nfabs; ++component) {
        const int offset = 2 * ndims * component;
        amrex::IntVect small, big;
        for (int d = 0; d < ndims; ++d) {
          small[d] = chunk_infos.at(offset + 0 * ndims + ndims - 1 - d);
          big[d] = chunk_infos.at(offset + 1 * ndims + ndims - 1 - d);
        }
        levboxes.at(component) = amrex::Box(small, big - 1);
      }

      // Don't set coarse level domain; this is already set by the driver
      if (level > 0) {
        amrex::Geometry geom = patchdata.amrcore->Geom(level - 1);
        geom.refine({2, 2, 2});
        patchdata.amrcore->SetGeometry(level, geom);
      }

      amrex::BoxList boxlist(std::move(levboxes));
      amrex::BoxArray boxarray(std::move(boxlist));
      patchdata.amrcore->SetBoxArray(level, boxarray);

      amrex::DistributionMapping dm(boxarray);
      patchdata.amrcore->SetDistributionMap(level, dm);

      patchdata.amrcore->SetupLevel(level, boxarray, dm,
                                    []() { return "Recovering"; });
    } // for level
  } // for patch
}

void carpetx_openpmd_t::InputOpenPMD(const cGH *const cctkGH,
                                     const std::vector<bool> &input_group,
                                     const std::string &input_dir,
                                     const std::string &input_file,
                                     const io_mode mode) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const bool is_checkpoint = mode == io_mode::checkpoint;

  // Set up timers
  static Timer timer("InputOpenPMD");
  Interval interval(timer);

  if (std::count(input_group.begin(), input_group.end(), true) == 0)
    return;

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD...");

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  if (is_root) {
    CCTK_VINFO("  openPMD input for groups:");
    for (int gi = 0; gi < CCTK_NumGroups(); ++gi)
      if (input_group.at(gi))
        CCTK_VINFO("    %s", CCTK_FullGroupName(gi));
  }

  if (!series) {
    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    const openPMD::Format format = get_format();
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << input_dir << "/" << input_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << input_dir << "/" << input_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    series = std::make_optional<openPMD::Series>(
        *filename, openPMD::Access::READ_ONLY, MPI_COMM_WORLD, options);
    assert(series->iterations.count(cctkGH->cctk_iteration));
    read_iter = std::make_optional<openPMD::Iteration>(
        series->iterations[cctkGH->cctk_iteration]);
  }
  assert(filename);
  assert(series);
  // assert(read_iters);
  assert(read_iter);

  CCTK_VINFO("  iteration: %d", cctkGH->cctk_iteration);

  const CCTK_REAL time = read_iter->time<CCTK_REAL>();
  const CCTK_REAL dt = read_iter->dt<CCTK_REAL>();
  const double timeUnitSI = read_iter->timeUnitSI();
  CCTK_VINFO("  time: %f", double(time));
  CCTK_VINFO("  dt: %f", double(dt));
  CCTK_VINFO("  time unit SI: %f", timeUnitSI);

  openPMD::Container<openPMD::Mesh> &meshes = read_iter->meshes;
  CCTK_VINFO("  found %d meshes", int(meshes.size()));

  if (io_verbose) {
    for (auto mesh_iter = meshes.begin(); mesh_iter != meshes.end();
         ++mesh_iter) {
      const std::string &mesh_name = mesh_iter->first;
      openPMD::Mesh &mesh = mesh_iter->second;
      CCTK_VINFO("    mesh: %s", mesh_name.c_str());

      const openPMD::Mesh::Geometry geometry = mesh.geometry();
      const std::vector<std::string> axisLabels = mesh.axisLabels();
      const std::vector<CCTK_REAL> gridSpacing = mesh.gridSpacing<CCTK_REAL>();
      const std::vector<double> gridGlobalOffset = mesh.gridGlobalOffset();
      const double gridUnitSI = mesh.gridUnitSI();
      const std::array<double, 7> unitDimension = mesh.unitDimension();
      const CCTK_REAL timeOffset = mesh.timeOffset<CCTK_REAL>();
      CCTK_VINFO("      geometry: %s", Geometry_string(geometry).c_str());
      CCTK_VINFO("      axis labels: %s",
                 vector_string(reversed(axisLabels)).c_str());
      CCTK_VINFO("      grid spacing: %s",
                 vector_string(reversed(gridSpacing)).c_str());
      CCTK_VINFO("      grid global offset: %s",
                 vector_string(reversed(gridGlobalOffset)).c_str());
      CCTK_VINFO("      grid unit SI: %f", gridUnitSI);
      CCTK_VINFO("      unit dimension: %s",
                 UnitDimension_string(unitDimension).c_str());
      CCTK_VINFO("      time offset: %f", double(timeOffset));

      CCTK_VINFO("      found %d components", int(mesh.size()));
      openPMD::Extent extent;
      for (auto record_component_iter = mesh.begin();
           record_component_iter != mesh.end(); ++record_component_iter) {
        const std::string &record_component_name = record_component_iter->first;
        openPMD::MeshRecordComponent &record_component =
            record_component_iter->second;
        CCTK_VINFO("        component: %s", record_component_name.c_str());

        const std::vector<CCTK_REAL> position =
            record_component.position<CCTK_REAL>();
        CCTK_VINFO("          position: %s",
                   vector_string(reversed(position)).c_str());

        const int ndims = record_component.getDimensionality();
        if (extent.empty())
          extent = record_component.getExtent();
        else
          assert(extent == record_component.getExtent());
        CCTK_VINFO("          ndims: %d", ndims);
        CCTK_VINFO("          extent: %s",
                   vector_string(reversed(extent)).c_str());

        const std::vector<openPMD::WrittenChunkInfo> chunks =
            record_component.availableChunks();
        CCTK_VINFO("          found %d chunks", int(chunks.size()));
        if (mesh_iter == meshes.begin() &&
            record_component_iter == mesh.begin()) {
          for (std::size_t n = 0; n < chunks.size(); ++n) {
            const openPMD::WrittenChunkInfo &chunk = chunks.at(n);
            CCTK_VINFO("            chunk: %d   start: %s   count: %s", int(n),
                       vector_string(reversed(chunk.offset)).c_str(),
                       vector_string(reversed(chunk.extent)).c_str());
          }
        }

      } // for record_component
    } // for mesh
  }

  // Post-read tasks
  std::vector<std::function<void()> > tasks;

  // First read grid functions in a loop over patches and levels

  // Loop over patches
  for (const auto &patchdata : ghext->patchdata) {
    // Loop over levels
    for (const auto &leveldata : patchdata.leveldata) {
      if (io_verbose)
        CCTK_VINFO("Reading patch %d level %d", patchdata.patch,
                   leveldata.level);

      // Determine grid structure

      const int *const nghosts = cctkGH->cctk_nghostzones;
      const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
      const amrex::Real *const xlo = geom.ProbLo();
      const amrex::Real *const xhi = geom.ProbHi();
      const amrex::Real *const dx = geom.CellSize();
      const box_t<CCTK_REAL, 3> rdomain{
          .lo = {xlo[0] - input_ghosts * nghosts[0] * dx[0],
                 xlo[1] - input_ghosts * nghosts[1] * dx[1],
                 xlo[2] - input_ghosts * nghosts[2] * dx[2]},
          .hi = {xhi[0] + input_ghosts * nghosts[0] * dx[0],
                 xhi[1] + input_ghosts * nghosts[1] * dx[1],
                 xhi[2] + input_ghosts * nghosts[2] * dx[2]}};
      const amrex::Box &dom = geom.Domain();
      const amrex::IntVect &ilo = dom.smallEnd();
      const amrex::IntVect &ihi = dom.bigEnd();
      // The domain is always vertex centred. The tensor components are
      // then staggered if necessary.
      const box_t<int, 3> idomain{
          .lo = {ilo[0] - input_ghosts * nghosts[0],
                 ilo[1] - input_ghosts * nghosts[1],
                 ilo[2] - input_ghosts * nghosts[2]},
          .hi = {ihi[0] + input_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + input_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + input_ghosts * nghosts[2] + 1 + 1}};
      if (io_verbose) {
        CCTK_VINFO("Level: %d", leveldata.level);
        CCTK_VINFO("  xmin: [%f,%f,%f]", double(rdomain.lo[0]),
                   double(rdomain.lo[1]), double(rdomain.lo[2]));
        CCTK_VINFO("  xmax: [%f,%f,%f]", double(rdomain.hi[0]),
                   double(rdomain.hi[1]), double(rdomain.hi[2]));
        CCTK_VINFO("  imin: [%d,%d,%d]", int(idomain.lo[0]), int(idomain.lo[1]),
                   int(idomain.lo[2]));
        CCTK_VINFO("  imax: [%d,%d,%d]", int(idomain.hi[0]), int(idomain.hi[1]),
                   int(idomain.hi[2]));
      }

      const int numgroups = CCTK_NumGroups();
      for (int gi = 0; gi < numgroups; ++gi) {
        if (input_group.at(gi)) {

          // Check group properties

          cGroup cgroup;
          const int ierr = CCTK_GroupData(gi, &cgroup);
          assert(!ierr);
          if (cgroup.grouptype != CCTK_GF)
            continue;
          assert(vartype_is_supported_real(cgroup.vartype));
          assert(cgroup.dim == 3);
          // cGroupDynamicData cgroupdynamicdata;
          // ierr = CCTK_GroupDynamicData(cctkGH, gi, &cgroupdynamicdata);
          // assert(!ierr);
          // TODO: Check whether group has storage
          // TODO: Check whether data are valid

          if (io_verbose)
            CCTK_VINFO("Reading group %s...", CCTK_FullGroupName(gi));

          auto &groupdata = *leveldata.groupdata.at(gi);
          // const int firstvarindex = groupdata.firstvarindex;
          const int numvars = groupdata.numvars;
          const int tl = 0;
          const bool group_has_no_ghosts = groupdata.nghostzones[0] == 0 &&
                                           groupdata.nghostzones[1] == 0 &&
                                           groupdata.nghostzones[2] == 0;

          // The body below is generic in the grid function group's storage
          // precision T (CCTK_REAL for REAL/REAL8 groups, float for REAL4
          // groups); `mfab` is either an amrex::MultiFab or an
          // amrex::fMultiFab, matching the group's AnyMultiFab alternative.
          const auto read_group = [&](auto &mfab) {
          using T = typename std::decay_t<decltype(mfab)>::value_type;
          const amrex::IndexType &indextype = mfab.ixType();
          const Arith::vect<bool, 3> is_cell_centred{indextype.cellCentered(0),
                                                     indextype.cellCentered(1),
                                                     indextype.cellCentered(2)};
          (void)is_cell_centred;

          const int num_local_components = mfab.local_size();

          // Read mesh

          const std::string meshname =
              make_meshname(gi, leveldata.patch, leveldata.level);
          if (io_verbose)
            CCTK_VINFO("Reading mesh %s...", meshname.c_str());
          assert(read_iter->meshes.count(meshname));
          const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);
          // TODO: The openPMD standard says to add an attribute
          // `refinementRatio`, which is a vector of integers

          // D6/HIGH: a REAL2 mesh is a Datatype::FLOAT record in *both*
          // modes (a checkpoint's raw16-in-f32 "carrier" and a viz file's
          // genuinely-widened float32 both determineDatatype<float>()), so
          // nothing about the record itself distinguishes them -- reading
          // one as the other silently produces zeros/garbage (see
          // real2_encoding_attribute_name's comment in io_real2.hxx).
          // Check the encoding attribute explicitly instead.
          if (vartype_is_real2(cgroup.vartype)) {
            std::string encoding = real2_encoding_f32_widened;
            if (mesh.containsAttribute(real2_encoding_attribute_name))
              encoding = mesh.getAttribute(real2_encoding_attribute_name)
                             .get<std::string>();
            const std::string expected = is_checkpoint
                                             ? real2_encoding_raw16_in_f32_carrier
                                             : real2_encoding_f32_widened;
            if (encoding != expected)
              CCTK_VERROR(
                  "openPMD mesh \"%s\" (group %s) in file \"%s\" was written "
                  "with CCTK_REAL2 encoding \"%s\", but is being read in "
                  "%s mode (expected \"%s\"). A CCTK_REAL2 checkpoint must "
                  "be read via CarpetX::recover_method, not "
                  "CarpetX::filereader_method or a plain openPMD viz read, "
                  "and vice versa.",
                  meshname.c_str(), CCTK_FullGroupName(gi), input_file.c_str(),
                  encoding.c_str(), is_checkpoint ? "checkpoint" : "viz",
                  expected.c_str());
          }

          // Define tensor components

          // TODO: Set component names according to the tensor type
          std::vector<openPMD::MeshRecordComponent> record_components;
          record_components.reserve(numvars);
          openPMD::Extent extent;
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string componentname = make_componentname(gi, vi);
            assert(mesh.count(componentname));
            record_components.push_back(mesh.at(componentname));
            const openPMD::MeshRecordComponent &record_component =
                record_components.back();
            if (vi == 0)
              extent = record_component.getExtent();
            else
              assert(extent == record_component.getExtent());
          }
          assert(int(record_components.size()) == numvars);

          // Read data

          if (io_verbose)
            CCTK_VINFO("Reading %d variables with %d components...", numvars,
                       num_local_components);

          // Loop over components (AMReX boxes)
          for (int local_component = 0; local_component < num_local_components;
               ++local_component) {
            const int component = mfab.IndexArray().at(local_component);

            const amrex::Box &fabbox =
                mfab.fabbox(component); // exterior (with ghosts)
            const box_t<int, 3> extbox{
                .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                       fabbox.smallEnd(2)},
                .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                       fabbox.bigEnd(2) + 1}};
            const amrex::Box &validbox =
                mfab.box(component); // interior (without ghosts)
            const box_t<int, 3> intbox{
                .lo = {validbox.smallEnd(0), validbox.smallEnd(1),
                       validbox.smallEnd(2)},
                .hi = {validbox.bigEnd(0) + 1, validbox.bigEnd(1) + 1,
                       validbox.bigEnd(2) + 1}};
            const box_t<int, 3> &box = input_ghosts ? extbox : intbox;

            const openPMD::Offset start =
                to_vector(reversed(box.lo - idomain.lo));
            const openPMD::Extent count = to_vector(reversed(box.shape()));
            const int np = box.size();
            assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
            for (int d = 0; d < 3; ++d)
              // assert(start.at(d) >= 0);
              assert(
                  start.at(d) <
                  std::numeric_limits<
                      std::remove_reference_t<decltype(start.at(d))> >::max() /
                      2);
            for (int d = 0; d < 3; ++d)
              assert(start.at(d) + count.at(d) <= extent.at(d));

            auto &fab = mfab[component];
            for (int vi = 0; vi < numvars; ++vi) {

              if (input_ghosts || intbox == extbox) {
                T *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                record_components.at(vi).loadChunkRaw(ptr, start, count);
#else
                record_components.at(vi).loadChunk(openPMD::shareRaw(ptr),
                                                   start, count);
#endif

              } else {
                const int amrex_size = extbox.size();
                T *const amrex_var_ptr = fab.dataPtr() + vi * amrex_size;
                const Arith::vect<int, 3> amrex_shape = extbox.shape();
                const Arith::vect<int, 3> amrex_offset = box.lo - extbox.lo;
                constexpr int amrex_di = 1;
                const int amrex_dj = amrex_di * amrex_shape[0];
                const int amrex_dk = amrex_dj * amrex_shape[1];
                // const int amrex_np = amrex_dk * amrex_shape[2];
                T *const amrex_ptr = amrex_var_ptr + amrex_di * amrex_offset[0] +
                                     amrex_dj * amrex_offset[1] +
                                     amrex_dk * amrex_offset[2];
                const Arith::vect<int, 3> contig_shape = box.shape();
                constexpr int contig_di = 1;
                const int contig_dj = contig_di * contig_shape[0];
                const int contig_dk = contig_dj * contig_shape[1];
                const int contig_np = contig_dk * contig_shape[2];
                assert(contig_np == np);
                T *const contig_ptr =
                    amrex_var_ptr + extbox.size() - box.size();
                // TODO: optimize memory layout
                // `T` can be `unsigned short` here: read_group is also
                // instantiated for the raw 16-bit checkpoint payload of a
                // CCTK_REAL2 group (see rawify_real2/derawify_real2 in
                // io_real2.hxx). There is no poison_value_t<unsigned short>
                // specialization (ipoison_t is only specialized for the
                // real/complex/int vartypes' actual element types, not for
                // this raw bit-pattern buffer type), so skip poisoning for
                // T=unsigned short rather than instantiate that template.
                // This buffer is only ever a temporary that
                // derawify_real2() reinterprets straight back into
                // CCTK_REAL2 immediately after this function returns, so
                // there is no separate "REAL2 poison pattern" to apply to
                // it here; poisoning of not-actually-read REAL2 ghost/
                // exterior points is instead handled by whichever CarpetX
                // code path initializes/poisons a freshly allocated
                // CCTK_REAL2 group in the first place (this function only
                // ever reads a checkpoint's saved interior).
                if constexpr (!std::is_same_v<T, unsigned short>) {
                  if (poison_undefined_values) {
                    const poison_value_t<T> poison_value;
#pragma omp simd
                    for (int n = 0; n < np; ++n)
                      poison_value.set_to_poison(contig_ptr[n]);
                  }
                }
                record_components.at(vi).loadChunkRaw(contig_ptr, start, count);
                tasks.emplace_back([=]() {
                  for (int k = 0; k < contig_shape[2]; ++k)
                    for (int j = 0; j < contig_shape[1]; ++j)
                      for (int i = 0; i < contig_shape[0]; ++i)
                        amrex_ptr[amrex_di * i + amrex_dj * j + amrex_dk * k] =
                            contig_ptr[contig_di * i + contig_dj * j +
                                       contig_dk * k];
                });
              }

            } // for vi
          } // for local_component
          }; // read_group

          if (vartype_is_real2(cgroup.vartype)) {
#ifdef HAVE_CCTK_REAL2
            // D6: CCTK_REAL2 groups are read either as their raw 16-bit
            // checkpoint payload (bit-exact round trip) or, for a regular
            // (non-checkpoint) file, as widened float32 data -- mirroring
            // how OutputOpenPMD below wrote them. Either way, the actual
            // read happens into a same-shaped temporary buffer (matching
            // whichever dtype the file actually holds), which is then
            // converted back into the group's real hMultiFab storage.
            //
            // The checkpoint leg reads a float32 "carrier" (see
            // widen_raw16_to_carrier/narrow_carrier_to_raw16 in
            // io_real2.hxx for why: this openPMD-api/ADIOS2 combination
            // cannot rediscover a multi-rank USHORT record on read), then
            // truncates it back down to the raw 16-bit payload before
            // reinterpreting that as CCTK_REAL2.
            //
            // The temporary buffer (carrier/widened) must stay alive past
            // this point: when the group has ghost zones, read_group's
            // "else" branch above only *queues* (into `tasks`) the work
            // that scatters this rank's read data into the buffer's final
            // in-memory layout -- the actual scatter, and hence the buffer
            // having its true contents, only happens once every group's
            // tasks have been queued and `series->flush()` runs the
            // deferred reads, much further down in this function. So the
            // buffer -> CCTK_REAL2 conversion cannot happen here either
            // (it would read stale/uninitialized data); it must be queued
            // as its own task, ordered after read_group's own tasks for
            // this group (guaranteed, since `tasks` is append-only and
            // read_group already pushed its tasks synchronously above).
            // The buffer is kept alive by moving it into that task's
            // closure (via shared_ptr, for std::function's copyability).
            hMultiFab &real2_mfab = std::get<hMultiFab>(*groupdata.mfab[tl]);
            // D6/MEDIUM: openPMD only fills the interior of this buffer
            // (input_ghosts is constexpr false, see `box` above), but the
            // narrow/derawify conversion queued below copies the *whole*
            // fab -- including ghost/exterior points -- into real2_mfab.
            // alloc_like_real2's contents are otherwise unspecified (freshly
            // arena'd memory), so pre-fill those not-actually-read points
            // with a recognisable pattern: the REAL2 poison value when
            // CarpetX::poison_undefined_values is set (matching what every
            // other freshly allocated CCTK_REAL2 group would already carry
            // there), else zero. setVal runs on the fab's own arena/device,
            // so this works for a GPU build too.
            if (is_checkpoint) {
              auto carrier = std::make_shared<amrex::fMultiFab>(
                  alloc_like_real2<amrex::fMultiFab>(real2_mfab));
              carrier->setVal(poison_undefined_values ? real2_poison_as_carrier()
                                                      : 0.0f);
              read_group(*carrier);
              tasks.emplace_back([carrier, &real2_mfab]() {
                const rawMultiFab raw = narrow_carrier_to_raw16(*carrier);
                derawify_real2(raw, real2_mfab);
              });
            } else {
              auto widened = std::make_shared<amrex::fMultiFab>(
                  alloc_like_real2<amrex::fMultiFab>(real2_mfab));
              widened->setVal(poison_undefined_values ? real2_poison_as_float()
                                                      : 0.0f);
              read_group(*widened);
              tasks.emplace_back([widened, &real2_mfab]() {
                narrow_float_to_real2(*widened, real2_mfab);
              });
            }
#else
            assert(0 && "unreachable: vartype_is_real2 is always false "
                        "without HAVE_CCTK_REAL2");
#endif
          } else if (vartype_is_real4(cgroup.vartype))
            read_group(std::get<amrex::fMultiFab>(*groupdata.mfab[tl]));
          else
            read_group(as_mfab_real(*groupdata.mfab[tl]));

          // Mark read variables as valid
          for (int vi = 0; vi < numvars; ++vi)
            groupdata.valid.at(tl).at(vi).set_all(
                input_ghosts || group_has_no_ghosts ? make_valid_all()
                                                    : make_valid_int(),
                []() { return "read from openPMD file"; });
        }
      } // for gi

    } // for leveldata
  } // for patchdata

  // Next read grid scalars and grid arrays

  {
    const int numgroups = CCTK_NumGroups();
    for (int gi = 0; gi < numgroups; ++gi) {
      if (input_group.at(gi)) {

        // Check group properties

        cGroup cgroup;
        const int ierr = CCTK_GroupData(gi, &cgroup);
        assert(!ierr);
        if (cgroup.grouptype == CCTK_GF)
          continue;
        assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
        assert(cgroup.dim >= 0);
        assert(cgroup.dim <= 3);

        if (io_verbose)
          CCTK_VINFO("Reading group %d %s...", gi, CCTK_FullGroupName(gi));

        auto &groupdata = *ghext->globaldata.arraygroupdata.at(gi);
        // const int firstvarindex = groupdata.firstvarindex;
        const int numvars = groupdata.numvars;
        const int tl = 0;

        // Determine grid structure

        using ivect = Arith::vect<int, dim>;

        const box_t<int, 3> idomain{.lo = ivect{0, 0, 0},
                                    .hi = ivect(groupdata.gsh)};

        // Read mesh

        const std::string meshname = make_meshname(gi, -1, -1);
        assert(read_iter->meshes.count(meshname));
        const openPMD::Mesh &mesh = read_iter->meshes.at(meshname);
        // TODO: The openPMD standard says to add an attribute
        // `refinementRatio`, which is a vector of integers

        // D6: unlike the grid function mesh, a REAL2 grid scalar/array's
        // openPMD Datatype itself already differs by mode (USHORT for a
        // checkpoint vs. FLOAT for viz), so the actual per-component read
        // below (loadChunkRaw<unsigned short> vs. loadChunk<float>) already
        // fails loudly with openPMD's own "Type conversion during chunk
        // loading not yet implemented!" on a mode mismatch. Check the
        // attribute here too, for a clearer error before that point and
        // for consistency with the grid function mesh check above.
        if (vartype_is_real2(cgroup.vartype)) {
          std::string encoding = real2_encoding_f32_widened;
          if (mesh.containsAttribute(real2_encoding_attribute_name))
            encoding = mesh.getAttribute(real2_encoding_attribute_name)
                           .get<std::string>();
          const std::string expected = is_checkpoint
                                           ? real2_encoding_raw16_in_f32_carrier
                                           : real2_encoding_f32_widened;
          if (encoding != expected)
            CCTK_VERROR(
                "openPMD mesh \"%s\" (group %s) in file \"%s\" was written "
                "with CCTK_REAL2 encoding \"%s\", but is being read in %s "
                "mode (expected \"%s\"). A CCTK_REAL2 checkpoint must be "
                "read via CarpetX::recover_method, not "
                "CarpetX::filereader_method or a plain openPMD viz read, "
                "and vice versa.",
                meshname.c_str(), CCTK_FullGroupName(gi), input_file.c_str(),
                encoding.c_str(), is_checkpoint ? "checkpoint" : "viz",
                expected.c_str());
        }

        // Define tensor components

        // TODO: Set component names according to the tensor type
        std::vector<openPMD::MeshRecordComponent> record_components;
        record_components.reserve(numvars);
        openPMD::Extent extent;
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string componentname = make_componentname(gi, vi);
          assert(mesh.count(componentname));
          record_components.push_back(mesh.at(componentname));
          const openPMD::MeshRecordComponent &record_component =
              record_components.back();
          if (vi == 0)
            extent = record_component.getExtent();
          else
            assert(extent == record_component.getExtent());
        }
        assert(int(record_components.size()) == numvars);

        // Read data

        if (io_verbose)
          CCTK_VINFO("Reading %d variables...", numvars);

        const bool group_has_no_ghosts = groupdata.nghostzones[0] == 0 &&
                                         groupdata.nghostzones[1] == 0 &&
                                         groupdata.nghostzones[2] == 0;

        // exterior (with ghosts)
        for (int d = 0; d < dim; ++d)
          assert(groupdata.lsh[d] == groupdata.gsh[d]);
        assert(all(Arith::vect<int, dim>(groupdata.lsh) ==
                   Arith::vect<int, dim>(groupdata.gsh)));
        const box_t<int, 3> extbox{.lo = ivect{0, 0, 0},
                                   .hi = ivect(groupdata.lsh)};
        // interior (without ghosts)
        const box_t<int, 3> intbox{.lo = ivect(groupdata.nghostzones),
                                   .hi = ivect(groupdata.lsh) -
                                         ivect(groupdata.nghostzones)};
        // It seems that openPMD assumes that chunks do not have ghost zones
        assert(!input_ghosts);
        const box_t<int, 3> &box = input_ghosts ? extbox : intbox;

        const openPMD::Offset start = to_vector(reversed(box.lo - idomain.lo));
        const openPMD::Extent count = to_vector(reversed(box.shape()));
        const int np = box.size();
        assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
        for (int d = 0; d < 3; ++d)
          // assert(start.at(d) >= 0);
          assert(start.at(d) <
                 std::numeric_limits<
                     std::remove_reference_t<decltype(start.at(d))> >::max() /
                     2);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) + count.at(d) <= extent.at(d));

        const Arith::vect<int, 3> cactus_shape = extbox.shape();
        constexpr int cactus_di = 1;
        const int cactus_dj = cactus_di * cactus_shape[0];
        const int cactus_dk = cactus_dj * cactus_shape[1];
        const int cactus_np = cactus_dk * cactus_shape[2];
        assert(cactus_di > 0);
        assert(cactus_dj > 0);
        assert(cactus_dk > 0);
        assert(cactus_np > 0);
        assert(int(groupdata.data.at(tl).size()) == numvars * cactus_np);
        for (int vi = 0; vi < numvars; ++vi) {
          void *const cactus_var_ptr =
              groupdata.data.at(tl).data_at(vi * cactus_np);
          if (input_ghosts || intbox == extbox) {
#if !OPENPMDAPI_VERSION_GE(0, 15, 0)
#define loadChunkRaw(ptr, start, count)                                        \
  loadChunk(openPMD::shareRaw(ptr), start, count)
#endif
            switch (cgroup.vartype) {
            case CCTK_VARIABLE_INT:
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_INT *>(cactus_var_ptr), start, count);
              break;
            case CCTK_VARIABLE_COMPLEX:
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_COMPLEX *>(cactus_var_ptr), start, count);
              break;
            case CCTK_VARIABLE_REAL4:
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_REAL4 *>(cactus_var_ptr), start, count);
              break;
#ifdef HAVE_CCTK_REAL2
            case CCTK_VARIABLE_REAL2:
              // Grid scalar/array I/O is always single-rank (unlike the
              // grid function path above), so, unlike CCTK_REAL2 grid
              // functions, there is no need for the float32 "carrier"
              // workaround for a checkpoint's raw 16-bit USHORT record
              // (see io_real2.hxx): read the raw bits directly into the
              // group's own CCTK_REAL2 storage -- bit-exact and zero-copy.
              // A regular (non-checkpoint, viz) file instead holds widened
              // float32 data (mirroring OutputOpenPMD below), so read it
              // into a temporary float buffer and narrow it back down to
              // CCTK_REAL2 once the deferred read actually runs (queued as
              // a `tasks` entry, run after series->flush() below -- the
              // same pattern the CCTK_REAL2 grid function path above uses).
              if (is_checkpoint) {
                record_components.at(vi).loadChunkRaw(
                    reinterpret_cast<unsigned short *>(cactus_var_ptr), start,
                    count);
              } else {
                auto floatbuf = std::shared_ptr<float[]>(new float[np]);
                record_components.at(vi).loadChunk(floatbuf, start, count);
                CCTK_REAL2 *const dst_ptr =
                    static_cast<CCTK_REAL2 *>(cactus_var_ptr);
                tasks.emplace_back([floatbuf, dst_ptr, np]() {
                  narrow_float_flat_to_real2(floatbuf.get(), dst_ptr, np);
                });
              }
              break;
#endif
            default:
              assert(vartype_is_real8(cgroup.vartype) &&
                    "Unexpected variable type");
              record_components.at(vi).loadChunkRaw(
                  static_cast<CCTK_REAL *>(cactus_var_ptr), start, count);
              break;
            }
          } else {
            auto cactus_ptr = &groupdata.data.at(tl);
            char *const cactus_var_ptr =
                static_cast<char *>(cactus_ptr->data_at(vi * cactus_np));
            const int vartypesize = CCTK_VarTypeSize(cgroup.vartype);
            const Arith::vect<int, 3> contig_shape = box.shape();
            constexpr int contig_di = 1;
            const int contig_dj = contig_di * contig_shape[0];
            const int contig_dk = contig_dj * contig_shape[1];
            const int contig_np = contig_dk * contig_shape[2];
            assert(contig_np == np);
            const int contig_offset = extbox.size() - box.size();
            // TODO: optimize memory layout
            const auto expand_box = [=](void *const contig_ptr) {
              for (int k = 0; k < contig_shape[2]; ++k)
                for (int j = 0; j < contig_shape[1]; ++j)
                  for (int i = 0; i < contig_shape[0]; ++i)
                    // TODO: copy whole contiguous strip at once
                    memcpy(cactus_var_ptr +
                               (cactus_di * i + cactus_dj * j + cactus_dk * k) *
                                   vartypesize,
                           cactus_var_ptr + (contig_di * i + contig_dj * j +
                                             contig_dk * k + contig_offset) *
                                                vartypesize,
                           vartypesize);
              if (poison_undefined_values) {
                const size_t typesize =
                    size_t(CCTK_VarTypeSize(cgroup.vartype));
                // TODO: Use AnyTypeScalarRef for this?
                std::vector<char> poison(typesize);
                assert(cgroup.vartype == CCTK_VARIABLE_INT ||
                       cgroup.vartype == CCTK_VARIABLE_COMPLEX ||
                       vartype_is_supported_real(cgroup.vartype));
                switch (cgroup.vartype) {
                case CCTK_VARIABLE_INT: {
                  poison_value_t<CCTK_INT> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
                case CCTK_VARIABLE_COMPLEX: {
                  poison_value_t<CCTK_COMPLEX> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
                case CCTK_VARIABLE_REAL4: {
                  poison_value_t<CCTK_REAL4> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
#ifdef HAVE_CCTK_REAL2
                case CCTK_VARIABLE_REAL2: {
                  poison_value_t<CCTK_REAL2> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
#endif
                default: {
                  assert(vartype_is_real8(cgroup.vartype));
                  poison_value_t<CCTK_REAL> poison_value;
                  poison_value.set_to_poison(poison.data(), 1);
                } break;
                }
                for (int k = extbox.lo[2]; k < extbox.hi[2]; ++k) {
                  for (int j = extbox.lo[1]; j < extbox.hi[1]; ++j) {
                    for (int i = extbox.lo[0]; i < extbox.hi[0]; ++i) {
                      const Arith::vect<int, dim> I{i, j, k};
                      if (any(I < box.lo || I >= box.hi))
                        memcpy(cactus_var_ptr + (cactus_di * i + cactus_dj * j +
                                                 cactus_dk * k) *
                                                    vartypesize,
                               poison.data(), poison.size());
                    }
                  }
                }
              }
            };
            assert(cgroup.vartype == CCTK_VARIABLE_INT ||
                   cgroup.vartype == CCTK_VARIABLE_COMPLEX ||
                   vartype_is_supported_real(cgroup.vartype));
            switch (cgroup.vartype) {
            case CCTK_VARIABLE_INT:
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_INT>(
                      static_cast<CCTK_INT *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_INT *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
            case CCTK_VARIABLE_COMPLEX:
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_COMPLEX>(
                      static_cast<CCTK_COMPLEX *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_COMPLEX *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
            case CCTK_VARIABLE_REAL4:
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_REAL4>(
                      static_cast<CCTK_REAL4 *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_REAL4 *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
#ifdef HAVE_CCTK_REAL2
            case CCTK_VARIABLE_REAL2:
              // Grid scalars/arrays never have ghost zones (nghostzones is
              // always 0, see SetupGlobals in driver.cxx), so intbox ==
              // extbox unconditionally and this whole branch is
              // unreachable for CCTK_ARRAY/CCTK_SCALAR groups (confirmed
              // by the assert(!input_ghosts) above). Only the checkpoint
              // (bit-exact raw 16-bit) case is implemented here, since its
              // source and destination element sizes match, so it fits the
              // existing generic (vartypesize-based) expand_box unchanged.
              // The viz (float32-widened) case would need a
              // differently-sized source buffer than the CCTK_REAL2-sized
              // destination, which expand_box's memcpy cannot bridge; it
              // is intentionally left unimplemented, as it can never run.
              assert(is_checkpoint &&
                    "REAL2 grid scalar/array viz (non-checkpoint) ghosted "
                    "read is unreachable: arrays/scalars have no ghost "
                    "zones");
              record_components.at(vi).loadChunk(
                  std::shared_ptr<unsigned short>(
                      static_cast<unsigned short *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](unsigned short *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
#endif
            default:
              assert(vartype_is_real8(cgroup.vartype));
              record_components.at(vi).loadChunk(
                  std::shared_ptr<CCTK_REAL>(
                      static_cast<CCTK_REAL *>(
                          cactus_ptr->data_at(contig_offset + cactus_np * vi)),
                      [=](CCTK_REAL *const ptr) {
                        expand_box(static_cast<void *>(ptr));
                      }),
                  start, count);
              break;
            }
          }

          // Mark read variables as valid
          groupdata.valid.at(tl).at(vi).set_all(
              input_ghosts || group_has_no_ghosts ? make_valid_all()
                                                  : make_valid_int(),
              []() { return "read from openPMD file"; });

        } // for vi
      }
    } // for gi
  }

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Performing all reads...");
  series->flush();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Post-processing data...");
  for (auto &task : tasks)
    std::move(task)();
  tasks.clear();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Closing iteration...");
  read_iter->close();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD: Deallocating objects...");
  read_iter.reset();
  series.reset();
  filename.reset();

  if (io_verbose)
    CCTK_VINFO("InputOpenPMD done.");

  if (io_verbose)
    timer.print();
}

////////////////////////////////////////////////////////////////////////////////

void carpetx_openpmd_t::OutputOpenPMD(const cGH *const cctkGH,
                                      const std::vector<bool> &output_group,
                                      const std::string &output_dir,
                                      const std::string &output_file,
                                      const io_mode mode) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const bool is_checkpoint = mode == io_mode::checkpoint;

  // Set up timers
  static Timer timer("OutputOpenPMD");
  Interval interval(timer);

  if (std::count(output_group.begin(), output_group.end(), true) == 0)
    return;

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD...");

  const openPMD::Format format = get_format();

  if (!series) {

    if (io_verbose)
      CCTK_VINFO("Creating openPMD object...");
    const int dir_mode = 0755;
    static std::once_flag create_directory;
    call_once(create_directory, [&]() {
      const int ierr = CCTK_CreateDirectory(dir_mode, output_dir.c_str());
      assert(ierr >= 0);
    });
    std::ostringstream buf;
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      buf << output_dir << "/" << output_file << ".it%08T"
          << openPMD::suffix(format);
      break;
    case openPMD::IterationEncoding::variableBased:
      buf << output_dir << "/" << output_file << openPMD::suffix(format);
      break;
    default:
      abort();
    }
    filename = std::make_optional<std::string>(buf.str());
    static bool is_first_output = true;
    const openPMD::Access access =
        iterationEncoding == openPMD::IterationEncoding::fileBased
            ? openPMD::Access::CREATE
        : iterationEncoding == openPMD::IterationEncoding::variableBased
            ? (is_first_output ? openPMD::Access::CREATE
                               : openPMD::Access::READ_WRITE)
            : openPMD::Access::READ_ONLY /*error*/;
    is_first_output = false;
    CCTK_VINFO("  options: %s", options.c_str());
    series = std::make_optional<openPMD::Series>(*filename, access,
                                                 MPI_COMM_WORLD, options);
    series->setIterationEncoding(iterationEncoding);

    {
      char const *const user = getenv("USER");
      if (user)
        series->setAuthor(user);
    }
    // Software is always "openPMD-api"
    // series->setSoftware("Einstein Toolkit <https://einsteintoolkit.org>");
    // series->setSoftwareVersion("...");
    // Date is set automatically
    // const std::time_t t = std::time(nullptr);
    // char date[100];
    // std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S",
    // std::localtime(&t)); series->setDate(date);
    // series->setSoftwareDependencies("...");
    {
      char hostname[1000];
      Util_GetHostName(hostname, sizeof hostname);
      series->setMachine(hostname);
    }

    write_iters =
        std::make_optional<openPMD::WriteIterations>(series->writeIterations());
  }
  assert(filename);
  assert(series);
  assert(write_iters);

  if (io_verbose)
    CCTK_VINFO("Creating iteration %d...", cctk_iteration);
  openPMD::Iteration write_iter = (*write_iters)[cctk_iteration];
  write_iter.setTime(cctk_time);
  write_iter.setDt(cctk_delta_time);
  write_iter.setTimeUnitSI(Unit::time);

  const int myproc = CCTK_MyProc(cctkGH);
  const int ioproc = 0;

  // Write parameters
  if (myproc == ioproc) {
    char *const data = IOUtil_GetAllParameters(cctkGH, 1 /*all*/);
    const std::string parameters(data);
    std::free(data);
    write_iter.setAttribute("AllParameters", parameters);
  }

  if (myproc == ioproc) {
    const int ndims = Loop::dim;
    write_iter.setAttribute<std::int64_t>("numDims", ndims);
    const int npatches = ghext->patchdata.size();
    write_iter.setAttribute<std::int64_t>("numPatches", npatches);
    std::vector<std::string> patch_suffixes(npatches);
    for (const auto &patchdata : ghext->patchdata) {
      const int patch = patchdata.patch;
      std::ostringstream buf;
      // String attributes cannot be empty; we thus always need to use a patch
      // suffix if (ghext->num_patches() > 1)
      buf << "_patch" << std::setw(2) << std::setfill('0') << patch;
      patch_suffixes.at(patch) = buf.str();
    }
    write_iter.setAttribute("patchSuffixes", patch_suffixes);
    for (const auto &patchdata : ghext->patchdata) {
      const int patch = patchdata.patch;
      const int nlevels = patchdata.leveldata.size();
      write_iter.setAttribute<std::int64_t>(
          "numLevels" + patch_suffixes.at(patch), nlevels);
      std::vector<std::string> level_suffixes(nlevels);
      for (const auto &leveldata : patchdata.leveldata) {
        const int level = leveldata.level;
        std::ostringstream buf;
        buf << patch_suffixes.at(patch) << "_lev" << std::setw(2)
            << std::setfill('0') << level;
        level_suffixes.at(level) = buf.str();
      }
      write_iter.setAttribute("levelSuffixes" + patch_suffixes.at(patch),
                              level_suffixes);
      for (const auto &leveldata : patchdata.leveldata) {
        const int level = leveldata.level;
        const amrex::FabArrayBase &mfab = *leveldata.fab;
        const int nchunks = mfab.size();
        std::vector<std::int64_t> chunk_infos(2 * ndims * nchunks);
        for (int component = 0; component < nchunks; ++component) {
          const amrex::Box &box = mfab.box(component);
          for (int d = 0; d < ndims; ++d) {
            chunk_infos.at(2 * ndims * component + 0 * ndims + ndims - 1 - d) =
                box.smallEnd()[d];
            chunk_infos.at(2 * ndims * component + 1 * ndims + ndims - 1 - d) =
                box.bigEnd()[d] + 1;
          }
        }
        write_iter.setAttribute("chunkInfo" + level_suffixes.at(level),
                                chunk_infos);
      }
    }
  } // if ioproc

  // First write grid functions in a loop over patches and levels

  // D6/HIGH: a REAL2 group's widened/carrier fMultiFab (built below by
  // widen_real2_to_float/widen_raw16_to_carrier) is a temporary passed
  // straight to write_group, which -- for the ghostless/non-ghosted branch
  // -- hands openPMD a raw pointer into it via storeChunkRaw. That is a
  // *deferred* write: openPMD::RecordComponent::storeChunkRaw requires the
  // buffer to outlive the next series->flush() (see the openPMD-api header
  // comment on that call), which for this function happens once, after
  // every patch/level/group has been queued below -- long after the
  // temporary bound to write_group's `const auto &mfab` parameter would
  // otherwise have been destroyed at the end of its full-expression. Keep
  // every such temporary alive here, across the whole loop, and only drop
  // them once flush() has actually run.
  std::vector<std::shared_ptr<amrex::fMultiFab> > real2_write_keepalive;

  // Loop over patches
  for (const auto &patchdata : ghext->patchdata) {
    // Loop over levels
    for (const auto &leveldata : patchdata.leveldata) {
      if (io_verbose)
        CCTK_VINFO("Writing patch %d level %d mesh...", patchdata.patch,
                   leveldata.level);

      // Determine grid structure

      const int *const nghosts = cctkGH->cctk_nghostzones;
      const amrex::Geometry &geom = patchdata.amrcore->Geom(leveldata.level);
      const amrex::Real *const xlo = geom.ProbLo();
      const amrex::Real *const xhi = geom.ProbHi();
      const amrex::Real *const dx = geom.CellSize();
      const box_t<CCTK_REAL, 3> rdomain{
          .lo = {xlo[0] - output_ghosts * nghosts[0] * dx[0],
                 xlo[1] - output_ghosts * nghosts[1] * dx[1],
                 xlo[2] - output_ghosts * nghosts[2] * dx[2]},
          .hi = {xhi[0] + output_ghosts * nghosts[0] * dx[0],
                 xhi[1] + output_ghosts * nghosts[1] * dx[1],
                 xhi[2] + output_ghosts * nghosts[2] * dx[2]}};
      const amrex::Box &dom = geom.Domain();
      const amrex::IntVect &ilo = dom.smallEnd();
      const amrex::IntVect &ihi = dom.bigEnd();
      // The domain is always vertex centred. The tensor components are
      // then staggered if necessary.
      const box_t<int, 3> idomain{
          .lo = {ilo[0] - output_ghosts * nghosts[0],
                 ilo[1] - output_ghosts * nghosts[1],
                 ilo[2] - output_ghosts * nghosts[2]},
          .hi = {ihi[0] + output_ghosts * nghosts[0] + 1 + 1,
                 ihi[1] + output_ghosts * nghosts[1] + 1 + 1,
                 ihi[2] + output_ghosts * nghosts[2] + 1 + 1}};
      if (io_verbose) {
        CCTK_VINFO("Patch: %d, Level: %d", patchdata.patch, leveldata.level);
        CCTK_VINFO("  xmin: [%f,%f,%f]", double(rdomain.lo[0]),
                   double(rdomain.lo[1]), double(rdomain.lo[2]));
        CCTK_VINFO("  xmax: [%f,%f,%f]", double(rdomain.hi[0]),
                   double(rdomain.hi[1]), double(rdomain.hi[2]));
        CCTK_VINFO("  imin: [%d,%d,%d]", int(idomain.lo[0]), int(idomain.lo[1]),
                   int(idomain.lo[2]));
        CCTK_VINFO("  imax: [%d,%d,%d]", int(idomain.hi[0]), int(idomain.hi[1]),
                   int(idomain.hi[2]));
      }

      // Create dataset

      const openPMD::Extent extent = to_vector(reversed(idomain.shape()));

      const int numgroups = CCTK_NumGroups();
      for (int gi = 0; gi < numgroups; ++gi) {
        if (output_group.at(gi)) {

          // Check group properties

          cGroup cgroup;
          const int ierr = CCTK_GroupData(gi, &cgroup);
          assert(!ierr);
          if (cgroup.grouptype != CCTK_GF)
            continue;
          assert(vartype_is_supported_real(cgroup.vartype));
          assert(cgroup.dim == 3);
          // cGroupDynamicData cgroupdynamicdata;
          // ierr = CCTK_GroupDynamicData(cctkGH, gi, &cgroupdynamicdata);
          // assert(!ierr);
          // TODO: Check whether group has storage
          // TODO: Check whether data are valid

          if (io_verbose)
            CCTK_VINFO("Writing group %d %s...", gi, CCTK_FullGroupName(gi));

          const auto &groupdata = *leveldata.groupdata.at(gi);
          // const int firstvarindex = groupdata.firstvarindex;
          const int numvars = groupdata.numvars;
          const int tl = 0;

          // The body below is generic in the grid function group's storage
          // precision T (CCTK_REAL for REAL/REAL8 groups, float for REAL4
          // groups); `mfab` is either an amrex::MultiFab or an
          // amrex::fMultiFab, matching the group's AnyMultiFab alternative.
          const auto write_group = [&](const auto &mfab) {
          using T = typename std::decay_t<decltype(mfab)>::value_type;
          const openPMD::Datatype datatype = openPMD::determineDatatype<T>();
          const openPMD::Dataset dataset(datatype, extent);

          const amrex::IndexType &indextype = mfab.ixType();
          const Arith::vect<bool, 3> is_cell_centred{indextype.cellCentered(0),
                                                     indextype.cellCentered(1),
                                                     indextype.cellCentered(2)};

          const int num_local_components = mfab.local_size();

          // Create mesh

          const std::string meshname =
              make_meshname(gi, leveldata.patch, leveldata.level);
          if (io_verbose)
            CCTK_VINFO("Defining mesh %s...", meshname.c_str());
          assert(!write_iter.meshes.contains(meshname));
          openPMD::Mesh mesh = write_iter.meshes[meshname];

          // D6/HIGH: record which CCTK_REAL2 encoding this mesh holds --
          // both the checkpoint carrier and the viz-widened data are plain
          // Datatype::FLOAT records with the same name, so this attribute
          // is the only on-disk way to tell them apart on read (see
          // real2_encoding_attribute_name's comment in io_real2.hxx).
          if (vartype_is_real2(cgroup.vartype))
            mesh.setAttribute(real2_encoding_attribute_name,
                              is_checkpoint
                                  ? real2_encoding_raw16_in_f32_carrier
                                  : real2_encoding_f32_widened);

          mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
          mesh.setAxisLabels(reversed(std::vector<std::string>{"x", "y", "z"}));
          mesh.setGridSpacing(to_vector<CCTK_REAL>(
              reversed(fmap([](auto x, auto y) { return x / CCTK_REAL(y); },
                            rdomain.hi - rdomain.lo, idomain.shape() - 1))));
          mesh.setGridGlobalOffset(to_vector<double>(reversed(rdomain.lo)));
          mesh.setGridUnitSI(Unit::length);
          // const std::map<openPMD::UnitDimension, double> unitDimension{
          //     {openPMD::UnitDimension::L, 1}};
          // mesh.setUnitDimension(unitDimension);
          mesh.setTimeOffset(CCTK_REAL(0)); // TODO: check interface.ccl

          // Cell centred grids are offset by 1/2
          const Arith::vect<double, 3> position =
              fmap([](auto c) { return 0.5 * c; }, is_cell_centred);

          // Define tensor components

          // TODO: Set component names according to the tensor type
          std::vector<openPMD::MeshRecordComponent> record_components;
          record_components.reserve(numvars);
#if 0
        switch (numvars) {
        case 1:
          for (int vi = 0; vi < numvars; ++vi) {
           record_omponents.push_back(mesh[openPMD::MeshRecordComponent::SCALAR]);
          }
          break;
        case 3:
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string cnames[] = {"x", "y", "z"};
            record_components.push_back(mesh[cnames[vi]]);
          }
          break;
        case 6:
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string cnames[] = {"xx", "xy", "xz", "yy", "yz", "zz"};
            record_components.push_back(mesh[cnames[vi]]);
          }
          break;
        default:
          CCTK_VWARN(CCTK_WARN_ALERT, "unsupported tensor type gi=%d group=%s",
                     gi, CCTK_FullGroupName(gi));
          for (int vi = 0; vi < numvars; ++vi) {
          const std::string varname = CCTK_VarName(firstvarindex + vi);
            CCTK_VINFO("Creating component %d %s", vi, varname.c_str());
            record_components.push_back(mesh[varname]);
          }
          break;
        }
#endif
          for (int vi = 0; vi < numvars; ++vi) {
            const std::string componentname = make_componentname(gi, vi);
            record_components.push_back(mesh[componentname]);
            auto &record_component = record_components.back();
            record_component.setPosition(to_vector<double>(reversed(position)));
          }
          assert(int(record_components.size()) == numvars);

          // Write data

          if (io_verbose)
            CCTK_VINFO("Writing %d variables with %d components...", numvars,
                       num_local_components);

          for (int vi = 0; vi < numvars; ++vi)
            record_components.at(vi).resetDataset(dataset);

          // Loop over components (AMReX boxes)
          for (int local_component = 0; local_component < num_local_components;
               ++local_component) {
            const int component = mfab.IndexArray().at(local_component);

            const amrex::Box &fabbox =
                mfab.fabbox(component); // exterior (with ghosts)
            const box_t<int, 3> extbox{
                .lo = {fabbox.smallEnd(0), fabbox.smallEnd(1),
                       fabbox.smallEnd(2)},
                .hi = {fabbox.bigEnd(0) + 1, fabbox.bigEnd(1) + 1,
                       fabbox.bigEnd(2) + 1}};
            const amrex::Box &validbox =
                mfab.box(component); // interior (without ghosts)
            const box_t<int, 3> intbox{
                .lo = {validbox.smallEnd(0), validbox.smallEnd(1),
                       validbox.smallEnd(2)},
                .hi = {validbox.bigEnd(0) + 1, validbox.bigEnd(1) + 1,
                       validbox.bigEnd(2) + 1}};
            // It seems that openPMD assumes that chunks do not have
            // ghost zones
            assert(!output_ghosts);
            const box_t<int, 3> &box = output_ghosts ? extbox : intbox;

            const openPMD::Offset start =
                to_vector(reversed(box.lo - idomain.lo));
            const openPMD::Extent count = to_vector(reversed(box.shape()));
            const int np = box.size();
            assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
            for (int d = 0; d < 3; ++d)
              // assert(start.at(d) >= 0);
              assert(
                  start.at(d) <
                  std::numeric_limits<
                      std::remove_reference_t<decltype(start.at(d))> >::max() /
                      2);
            for (int d = 0; d < 3; ++d)
              assert(start.at(d) + count.at(d) <= extent.at(d));

            const auto &fab = mfab[component];
            for (int vi = 0; vi < numvars; ++vi) {
              if (output_ghosts || intbox == extbox) {
                const T *const ptr = fab.dataPtr() + vi * np;
#if OPENPMDAPI_VERSION_GE(0, 15, 0)
                record_components.at(vi).storeChunkRaw(ptr, start, count);
#else
                record_components.at(vi).storeChunk(openPMD::shareRaw(ptr),
                                                    start, count);
#endif
              } else {
                std::shared_ptr<T> ptr(new T[np], std::default_delete<T[]>());
                const Arith::vect<int, 3> amrex_shape = extbox.shape();
                const Arith::vect<int, 3> amrex_offset = box.lo - extbox.lo;
                constexpr int amrex_di = 1;
                const int amrex_dj = amrex_di * amrex_shape[0];
                const int amrex_dk = amrex_dj * amrex_shape[1];
                const int amrex_np = amrex_dk * amrex_shape[2];
                const T *restrict const amrex_ptr =
                    fab.dataPtr() + vi * amrex_np + amrex_di * amrex_offset[0] +
                    amrex_dj * amrex_offset[1] + amrex_dk * amrex_offset[2];
                const Arith::vect<int, 3> contig_shape = box.shape();
                constexpr int contig_di = 1;
                const int contig_dj = contig_di * contig_shape[0];
                const int contig_dk = contig_dj * contig_shape[1];
                const int contig_np = contig_dk * contig_shape[2];
                assert(contig_np == np);
                T *restrict const contig_ptr = ptr.get();
                for (int k = 0; k < contig_shape[2]; ++k)
                  for (int j = 0; j < contig_shape[1]; ++j)
#pragma omp simd
                    for (int i = 0; i < contig_shape[0]; ++i)
                      contig_ptr[contig_di * i + contig_dj * j +
                                 contig_dk * k] =
                          amrex_ptr[amrex_di * i + amrex_dj * j + amrex_dk * k];
                record_components.at(vi).storeChunk(std::move(ptr), start,
                                                    count);
              }
            } // for vi
          } // for local_component
          }; // write_group

          if (vartype_is_real2(cgroup.vartype)) {
#ifdef HAVE_CCTK_REAL2
            // D6: widen to float32 for regular (non-checkpoint) output.
            // For a checkpoint, the raw 16-bit payload must round-trip
            // bit-exactly, but it cannot be written as a USHORT openPMD
            // record directly: this openPMD-api/ADIOS2 combination can't
            // rediscover a multi-rank USHORT record on read (see the
            // comment on widen_raw16_to_carrier/narrow_carrier_to_raw16 in
            // io_real2.hxx). So the raw 16 bits are zero-extended and
            // bit-punned into a float32 "carrier" and written through the
            // same (already load-bearing, proven-working) float32 code
            // path used for CCTK_REAL4 groups.
            const hMultiFab &real2_mfab =
                std::get<hMultiFab>(*groupdata.mfab[tl]);
            // D6/HIGH: keep the widened/carrier buffer alive past
            // series->flush() (see real2_write_keepalive's comment above)
            // -- write_group's ghostless/non-ghosted branch hands openPMD a
            // raw pointer into it for a deferred storeChunkRaw.
            const auto real2_buf = std::make_shared<amrex::fMultiFab>(
                is_checkpoint ? widen_raw16_to_carrier(rawify_real2(real2_mfab))
                              : widen_real2_to_float(real2_mfab));
            real2_write_keepalive.push_back(real2_buf);
            write_group(*real2_buf);
#else
            assert(0 && "unreachable: vartype_is_real2 is always false "
                        "without HAVE_CCTK_REAL2");
#endif
          } else if (vartype_is_real4(cgroup.vartype))
            write_group(std::get<amrex::fMultiFab>(*groupdata.mfab[tl]));
          else
            write_group(as_mfab_real(*groupdata.mfab[tl]));
        }
      } // for gi

    } // for leveldata
  } // for patchdata

  // Next write grid scalars and grid arrays

  if (myproc == ioproc) {
    const int numgroups = CCTK_NumGroups();
    for (int gi = 0; gi < numgroups; ++gi) {
      if (output_group.at(gi)) {

        // Check group properties

        cGroup cgroup;
        const int ierr = CCTK_GroupData(gi, &cgroup);
        assert(!ierr);
        if (cgroup.grouptype == CCTK_GF)
          continue;
        assert(cgroup.disttype == CCTK_DISTRIB_CONSTANT);
        assert(cgroup.dim >= 0);
        assert(cgroup.dim <= 3);

        if (io_verbose)
          CCTK_VINFO("Writing group %d %s...", gi, CCTK_FullGroupName(gi));

        const auto &groupdata = *ghext->globaldata.arraygroupdata.at(gi);
        // const int firstvarindex = groupdata.firstvarindex;
        const int numvars = groupdata.numvars;
        const int tl = 0;

        // Determine grid structure

        using ivect = Arith::vect<int, dim>;

        const box_t<int, 3> idomain{.lo = ivect{0, 0, 0},
                                    .hi = ivect(groupdata.gsh)};

        // Create dataset

        const openPMD::Datatype datatype = [is_checkpoint](int varType) {
          switch (varType) {
          case CCTK_VARIABLE_INT:
            return openPMD::determineDatatype<CCTK_INT>();
          case CCTK_VARIABLE_COMPLEX:
            return openPMD::determineDatatype<CCTK_COMPLEX>();
          case CCTK_VARIABLE_REAL4:
            return openPMD::determineDatatype<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
          case CCTK_VARIABLE_REAL2:
            // Checkpoints store CCTK_REAL2's raw 16-bit payload bit-exactly
            // (D6); a regular (viz) file instead widens to float32, since
            // openPMD has no native fp16 dtype (see io_real2.hxx).
            return is_checkpoint ? openPMD::determineDatatype<unsigned short>()
                                 : openPMD::determineDatatype<float>();
#endif
          default:
            assert(vartype_is_real8(varType) && "Unexpected varType");
            return openPMD::determineDatatype<CCTK_REAL>();
          }
        }(cgroup.vartype);
        const openPMD::Extent extent = to_vector(reversed(idomain.shape()));
        const openPMD::Dataset dataset(datatype, extent);

        // Create mesh

        const std::string meshname = make_meshname(gi, -1, -1);
        openPMD::Mesh mesh = write_iter.meshes[meshname];

        // Unlike the grid function mesh above, a REAL2 grid scalar/array's
        // openPMD *Datatype itself* already differs by mode (USHORT for a
        // checkpoint's raw bits vs. FLOAT for viz-widened data, see the
        // `datatype` lambda above), so a mode mismatch on read already
        // fails loudly with openPMD's own "Type conversion during chunk
        // loading not yet implemented!" std::runtime_error (RecordComponent
        // ::isSameFloatingPoint), unlike the GF mesh's ambiguous
        // both-modes-are-FLOAT case. The attribute is set anyway, for
        // uniformity with the GF mesh and in case that dtype distinction
        // is ever unified away.
        if (vartype_is_real2(cgroup.vartype))
          mesh.setAttribute(real2_encoding_attribute_name,
                            is_checkpoint ? real2_encoding_raw16_in_f32_carrier
                                          : real2_encoding_f32_widened);

        // mesh.setGeometry(openPMD::Mesh::Geometry::cartesian);
        // mesh.setAxisLabels(reversed(std::vector<std::string>{"x", "y",
        // "z"}));
        // mesh.setGridSpacing(to_vector<CCTK_REAL>(
        //     reversed(fmap([](auto x, auto y) { return x / CCTK_REAL(y); },
        //                   rdomain.hi - rdomain.lo, idomain.shape() - 1))));
        // mesh.setGridGlobalOffset(to_vector<double>(reversed(rdomain.lo)));
        // mesh.setGridUnitSI(Unit::length);
        // // const std::map<openPMD::UnitDimension, double> unitDimension{
        // //     {openPMD::UnitDimension::L, 1}};
        // // mesh.setUnitDimension(unitDimension);
        mesh.setTimeOffset(CCTK_REAL(0)); // TODO: check interface.ccl

        // // Cell centred grids are offset by 1/2
        // const Arith::vect<double, 3> position{0, 0, 0};

        // Define tensor components

        // TODO: Set component names according to the tensor type
        std::vector<openPMD::MeshRecordComponent> record_components;
        record_components.reserve(numvars);
        for (int vi = 0; vi < numvars; ++vi) {
          const std::string componentname = make_componentname(gi, vi);
          record_components.push_back(mesh[componentname]);
          // auto &record_component = record_components.back();
          // record_component.setPosition(to_vector<double>(reversed(position)));
        }
        assert(int(record_components.size()) == numvars);

        // Write data

        if (io_verbose)
          CCTK_VINFO("Writing %d variables...", numvars);

        for (int vi = 0; vi < numvars; ++vi)
          record_components.at(vi).resetDataset(dataset);

        // exterior (with ghosts)
        for (int d = 0; d < dim; ++d)
          assert(groupdata.lsh[d] == groupdata.gsh[d]);
        assert(all(Arith::vect<int, dim>(groupdata.lsh) ==
                   Arith::vect<int, dim>(groupdata.gsh)));
        const box_t<int, 3> extbox{.lo = ivect{0, 0, 0},
                                   .hi = ivect(groupdata.lsh)};
        // interior (without ghosts)
        const box_t<int, 3> intbox{.lo = ivect(groupdata.nghostzones),
                                   .hi = ivect(groupdata.lsh) -
                                         ivect(groupdata.nghostzones)};
        // It seems that openPMD assumes that chunks do not have ghost zones
        assert(!output_ghosts);
        const box_t<int, 3> &box = output_ghosts ? extbox : intbox;

        const openPMD::Offset start = to_vector(reversed(box.lo - idomain.lo));
        const openPMD::Extent count = to_vector(reversed(box.shape()));
        const int np = box.size();
        assert(int(count.at(0) * count.at(1) * count.at(2)) == np);
        for (int d = 0; d < 3; ++d)
          // assert(start.at(d) >= 0);
          assert(start.at(d) <
                 std::numeric_limits<
                     std::remove_reference_t<decltype(start.at(d))> >::max() /
                     2);
        for (int d = 0; d < 3; ++d)
          assert(start.at(d) + count.at(d) <= extent.at(d));

        const Arith::vect<int, 3> cactus_shape = extbox.shape();
        constexpr int cactus_di = 1;
        const int cactus_dj = cactus_di * cactus_shape[0];
        const int cactus_dk = cactus_dj * cactus_shape[1];
        const int cactus_np = cactus_dk * cactus_shape[2];
        assert(cactus_di > 0);
        assert(cactus_dj > 0);
        assert(cactus_dk > 0);
        assert(cactus_np > 0);
        assert(int(groupdata.data.at(tl).size()) == numvars * cactus_np);
        for (int vi = 0; vi < numvars; ++vi) {
          const void *const var_ptr =
              groupdata.data.at(tl).data_at(vi * cactus_np);
          if (output_ghosts || intbox == extbox) {
#if !OPENPMDAPI_VERSION_GE(0, 15, 0)
#define storeChunkRaw(ptr, start, count)                                       \
  storeChunk(openPMD::shareRaw(ptr), start, count)
#endif
            switch (cgroup.vartype) {
            case CCTK_VARIABLE_INT:
              record_components.at(vi).storeChunkRaw(
                  static_cast<CCTK_INT const *>(var_ptr), start, count);
              break;
            case CCTK_VARIABLE_COMPLEX:
              record_components.at(vi).storeChunkRaw(
                  static_cast<CCTK_COMPLEX const *>(var_ptr), start, count);
              break;
            case CCTK_VARIABLE_REAL4:
              record_components.at(vi).storeChunkRaw(
                  static_cast<CCTK_REAL4 const *>(var_ptr), start, count);
              break;
#ifdef HAVE_CCTK_REAL2
            case CCTK_VARIABLE_REAL2:
              // See the matching read-side comment above: grid
              // scalar/array I/O is always single-rank, so a checkpoint's
              // raw 16-bit payload can be written directly and bit-exactly
              // (no float32 "carrier" workaround needed, unlike the grid
              // function path). A regular (viz) file instead widens to
              // float32 (mirroring the datatype lambda above), via a
              // temporary buffer kept alive (through the deferred write,
              // past series->flush() below) by the shared_ptr storeChunk
              // overload's own internal reference, exactly like the dead
              // (but instructive) AnyTypeVector-buffer branch below.
              if (is_checkpoint) {
                record_components.at(vi).storeChunkRaw(
                    reinterpret_cast<unsigned short const *>(var_ptr), start,
                    count);
              } else {
                auto floatbuf = std::shared_ptr<float[]>(new float[np]);
                widen_real2_flat_to_float(static_cast<const CCTK_REAL2 *>(var_ptr),
                                          floatbuf.get(), np);
                record_components.at(vi).storeChunk(floatbuf, start, count);
              }
              break;
#endif
            default:
              assert(vartype_is_real8(cgroup.vartype) &&
                    "Unexpected variable type");
              record_components.at(vi).storeChunkRaw(
                  static_cast<CCTK_REAL const *>(var_ptr), start, count);
              break;
            }
          } else {
            auto cactus_ptr = &groupdata.data.at(tl);
            const Arith::vect<int, 3> contig_shape = box.shape();
            constexpr int contig_di = 1;
            const int contig_dj = contig_di * contig_shape[0];
            const int contig_dk = contig_dj * contig_shape[1];
            const int contig_np = contig_dk * contig_shape[2];
            assert(contig_np == np);
            auto contig_ptr =
                new GHExt::GlobalData::AnyTypeVector(cgroup.vartype, np);
            for (int k = 0; k < contig_shape[2]; ++k)
              for (int j = 0; j < contig_shape[1]; ++j)
                for (int i = 0; i < contig_shape[0]; ++i)
                  // TODO: copy whole contiguous strip at once
                  memcpy(contig_ptr->data_at(contig_di * i + contig_dj * j +
                                             contig_dk * k),
                         cactus_ptr->data_at(cactus_di * i + cactus_dj * j +
                                             cactus_dk * k + vi * cactus_np),
                         CCTK_VarTypeSize(cgroup.vartype));
            switch (cgroup.vartype) {
            case CCTK_VARIABLE_INT:
              record_components.at(vi).storeChunk(
                  std::shared_ptr<CCTK_INT>(
                      static_cast<CCTK_INT *>(contig_ptr->data_at(0)),
                      [=](CCTK_INT *const) { delete contig_ptr; }),
                  start, count);
              break;
            case CCTK_VARIABLE_COMPLEX:
              record_components.at(vi).storeChunk(
                  std::shared_ptr<CCTK_COMPLEX>(
                      static_cast<CCTK_COMPLEX *>(contig_ptr->data_at(0)),
                      [=](CCTK_COMPLEX *const) { delete contig_ptr; }),
                  start, count);
              break;
            case CCTK_VARIABLE_REAL4:
              record_components.at(vi).storeChunk(
                  std::shared_ptr<CCTK_REAL4>(
                      static_cast<CCTK_REAL4 *>(contig_ptr->data_at(0)),
                      [=](CCTK_REAL4 *) { delete contig_ptr; }),
                  start, count);
              break;
#ifdef HAVE_CCTK_REAL2
            case CCTK_VARIABLE_REAL2:
              // See the matching comment on the ghosted read-side branch
              // above: this whole branch is unreachable for
              // CCTK_ARRAY/CCTK_SCALAR groups (they have no ghost zones),
              // so only the checkpoint (bit-exact) case, whose source and
              // destination element sizes match, is implemented.
              assert(is_checkpoint &&
                    "REAL2 grid scalar/array viz (non-checkpoint) ghosted "
                    "write is unreachable: arrays/scalars have no ghost "
                    "zones");
              record_components.at(vi).storeChunk(
                  std::shared_ptr<unsigned short>(
                      static_cast<unsigned short *>(contig_ptr->data_at(0)),
                      [=](unsigned short *) { delete contig_ptr; }),
                  start, count);
              break;
#endif
            default:
              assert(vartype_is_real8(cgroup.vartype) &&
                    "Unexpected variable type");
              record_components.at(vi).storeChunk(
                  std::shared_ptr<CCTK_REAL>(
                      static_cast<CCTK_REAL *>(contig_ptr->data_at(0)),
                      [=](CCTK_REAL *) { delete contig_ptr; }),
                  start, count);
              break;
            }
          }
        } // for vi
      }
    }
  }

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD: Performing all writes...");
  series->flush();

  // Every deferred storeChunkRaw queued above has now actually run; the
  // REAL2 widened/carrier buffers kept alive for it may finally be freed.
  real2_write_keepalive.clear();

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD: Closing iteration...");
  write_iter.close();

  if (CCTK_MyProc(nullptr) == 0) {
    std::ostringstream buf;
    buf << output_dir << "/" << output_file << ".openpmd.visit";
    const std::string visitname = buf.str();
    std::ofstream visit(visitname, std::ios::app);
    assert(visit.good());
    switch (iterationEncoding) {
    case openPMD::IterationEncoding::fileBased:
      visit << output_file << ".it" << std::setw(8) << std::setfill('0')
            << cctk_iteration << openPMD::suffix(format) << "\n";
      break;
    case openPMD::IterationEncoding::variableBased:
      visit << output_file << openPMD::suffix(format) << "\n";
      break;
    default:
      abort();
    }
  }

  switch (iterationEncoding) {
  case openPMD::IterationEncoding::fileBased:
    write_iters.reset();
    series.reset();
    filename.reset();
    break;
  case openPMD::IterationEncoding::variableBased:
    // do nothing
    break;
  default:
    abort();
  }

  if (io_verbose)
    CCTK_VINFO("OutputOpenPMD done.");

  if (io_verbose)
    timer.print();
}

} // namespace CarpetX

#else

namespace CarpetX {
void ShutdownOpenPMD() {}
} // namespace CarpetX

#endif // #ifdef HAVE_CAPABILITY_OPENPMD
