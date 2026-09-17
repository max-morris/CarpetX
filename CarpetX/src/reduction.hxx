#ifndef CARPETX_CARPETX_REDUCTION_HXX
#define CARPETX_CARPETX_REDUCTION_HXX

#include "defs.hxx"
#include "vect.hxx"

#include <cctk.h>

#include <mpi.h>

#include <cmath>
#include <ostream>

namespace CarpetX {
using namespace Arith;

template <typename T, int D> struct reduction {
  // TODO: contains_inf, contains_nan?
  T min, max, sum, sum2;
  T vol, maxabs, sumabs, sum2abs;
  vect<T, D> minloc, maxloc, sumloc;

  // We currently omit minloc/maxloc/sumloc (TODO: fix this)
  using tuple_type = amrex::GpuTuple<T, T, T, T, T, T, T, T>;
  constexpr reduction(tuple_type);
  constexpr operator tuple_type() const;

  constexpr reduction();
  constexpr reduction(const vect<T, D> &p, const T &V, const T &x);
  constexpr reduction(const reduction &x, const reduction &y);
  constexpr reduction operator+(const reduction &y) const {
    return reduction(*this, y);
  }
  constexpr reduction &operator+=(const reduction &y) noexcept {
    return *this = *this + y;
  }

  constexpr T avg() const noexcept { return sum / vol; }
  constexpr T sdv() const noexcept {
    // Splitting pow2(sum/vol) improves floating-point accuracy
    // return sqrt(max1(T(0), sum2 / vol - pow2(sum / vol)));
    return sqrt(max1(T(0), sum2 / vol - pow2(sum) / pow2(vol)));
  }
  constexpr T norm0() const noexcept { return vol; }
  constexpr T norm1() const noexcept { return sumabs / vol; }
  constexpr T norm2() const noexcept { return sqrt(sum2abs / vol); }
  constexpr T norm_inf() const noexcept { return maxabs; }

  template <typename T1, int D1>
  friend std::ostream &operator<<(std::ostream &os,
                                  const reduction<T1, D1> &red);
};

template <typename T, int D>
constexpr reduction<T, D>::reduction(tuple_type tuple)
    : min(amrex::get<0>(tuple)), max(amrex::get<1>(tuple)),
      sum(amrex::get<2>(tuple)), sum2(amrex::get<3>(tuple)),
      vol(amrex::get<4>(tuple)), maxabs(amrex::get<5>(tuple)),
      sumabs(amrex::get<6>(tuple)), sum2abs(amrex::get<7>(tuple)), minloc{},
      maxloc{}, sumloc{} {}

template <typename T, int D>
constexpr reduction<T, D>::operator reduction<T, D>::tuple_type() const {
  return tuple_type{min, max, sum, sum2, vol, maxabs, sumabs, sum2abs};
}

template <typename T, int D>
constexpr reduction<T, D>::reduction()
    : min(1.0 / 0.0), max(-1.0 / 0.0), sum(0.0), sum2(0.0), vol(0.0),
      maxabs(0.0), sumabs(0.0), sum2abs(0.0),
      minloc(vect<T, D>::pure(0.0 / 0.0)), maxloc(vect<T, D>::pure(0.0 / 0.0)),
      sumloc(vect<T, D>::pure(0.0)) {}

template <typename T, int D>
constexpr reduction<T, D>::reduction(const vect<T, D> &p, const T &V,
                                     const T &x)
    : min(x), max(x), sum(V * x), sum2(V * pow2(x)), vol(V), maxabs(fabs(x)),
      sumabs(V * fabs(x)), sum2abs(V * pow2(fabs(x))), minloc(p), maxloc(p),
      sumloc(x * p) {}

template <typename T, int D>
constexpr reduction<T, D>::reduction(const reduction &x, const reduction &y)
    : min(min1(x.min, y.min)), max(max1(x.max, y.max)), sum(x.sum + y.sum),
      sum2(x.sum2 + y.sum2), vol(x.vol + y.vol),
      maxabs(max1(x.maxabs, y.maxabs)), sumabs(x.sumabs + y.sumabs),
      sum2abs(x.sum2abs + y.sum2abs),
      minloc(x.min <= y.min ? x.minloc : y.minloc),
      maxloc(x.max >= y.max ? x.maxloc : y.maxloc),
      sumloc(x.sumloc + y.sumloc) {}

template <typename T, int D>
std::ostream &operator<<(std::ostream &os, const reduction<T, D> &red) {
  return os << "reduction{\n"
            << "  min:      " << red.min << "\n"
            << "  max:      " << red.max << "\n"
            << "  sum:      " << red.sum << "\n"
            << "  sum2:     " << red.sum2 << "\n"
            << "  vol:      " << red.vol << "\n"
            << "  maxabs:   " << red.maxabs << "\n"
            << "  sumabs:   " << red.sumabs << "\n"
            << "  sum2abs:  " << red.sum2abs << "\n"
            << "  minloc:   " << red.minloc << "\n"
            << "  maxloc:   " << red.maxloc << "\n"
            << "  sumloc:   " << red.sumloc << "\n"
            << "  avg:      " << red.avg() << "\n"
            << "  sdv:      " << red.sdv() << "\n"
            << "  norm1:    " << red.norm1() << "\n"
            << "  norm2:    " << red.norm2() << "\n"
            << "  norm_inf: " << red.norm_inf() << "\n"
            << "}\n";
}

template <typename T, int D> struct combine {
  using value_type = typename reduction<T, D>::tuple_type;
  constexpr AMREX_GPU_DEVICE value_type init() const {
    return (value_type)reduction<T, D>();
  }
  constexpr AMREX_GPU_DEVICE value_type operator()(const value_type &x,
                                                   const value_type &y) const {
    return (value_type)reduction<T, D>(x), reduction<T, D>(y);
  }
};

typedef reduction<CCTK_REAL, dim> reduction_CCTK_REAL;
#ifndef __NVCOMPILER
#pragma omp declare reduction(reduction:reduction_CCTK_REAL : omp_out += omp_in)
#endif

MPI_Datatype reduction_mpi_datatype_CCTK_REAL();
MPI_Op reduction_mpi_op();

// Reduce grid function variable `vi` of group `gi`, time level `tl`. Works
// for CCTK_REAL/REAL8 (double), CCTK_REAL4 (float), and CCTK_REAL2
// (_Float16, when HAVE_CCTK_REAL2) groups: REAL4/REAL2 source data is read
// in its native precision and accumulated (and MPI-reduced) in double, so
// the result type is always reduction<CCTK_REAL, dim> regardless of the
// group's storage precision (see reduce_typed<MF> in reduction.cxx).
reduction<CCTK_REAL, dim> reduce(int gi, int vi, int tl);

// Per-level statistics of one variable (out_norm_per_level): plain point
// averages over one refinement level's interior (no ghost zones), with the
// fp16 admissibility ratio over the part of that interior not covered by a
// finer level. See reduce_level in reduction.cxx.
struct level_reduction_t {
  CCTK_REAL sum2;         // sum of u^2 over the counted points
  CCTK_REAL maxabs;       // max |u|
  CCTK_REAL min, max;     // min and max of u
  long long npoints;      // points counted (interior, mask applied)
  long long nsubnormal16; // points with 0 < |u| < 2^-14 (fp16 min normal)
  // Fine-masked pass: max over points of q(|u|) / max(|D2u|, floor), and
  // the same restricted to the shell around out_norm_admiss_radius. -inf
  // when no point contributed (written as NaN).
  CCTK_REAL admiss, admiss_shell;
  bool has_admiss; // false when the group has no ghost zones (no D2u)

  CCTK_REAL norm2() const noexcept {
    return npoints > 0 ? sqrt(sum2 / CCTK_REAL(npoints)) : CCTK_REAL(0);
  }
};

// Reduce variable `vi` of grid function group `gi`, time level `tl`, on
// refinement level `level` only, MPI-reduced across all processes.
// `mask_vi` is the Cactus variable index of the mask variable, or -1 for no
// mask; the mask threshold and the admissibility constants are read from
// the out_norm_* parameters.
level_reduction_t reduce_level(int gi, int vi, int tl, int level, int mask_vi);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_REDUCTION_HXX
