#ifndef CARPETX_CARPETX_IO_REAL2_HXX
#define CARPETX_CARPETX_IO_REAL2_HXX

// Shared helpers for CCTK_REAL2 (binary16) grid function I/O, used by
// io_openpmd.cxx, io_silo.cxx, and io_adios2.cxx. (io_tsv.cxx needs no
// special handling: its std::visit-based per-group bodies already widen
// every element to CCTK_REAL for text formatting, regardless of storage
// precision, which already covers what D6 wants for that format.)
//
// Per the design's D6:
//
//   - Visualization output formats (openPMD/Silo/ADIOS2) have no native
//     fp16 dtype, so CCTK_REAL2 groups are *widened* to float32 for these
//     formats. A temporary amrex::fMultiFab (same box layout/component
//     count/ghost width as the source hMultiFab) is built holding the
//     widened/narrowed values, then handed to the exact same generic
//     (float-instantiated) read/write code path already used for
//     CCTK_REAL4 groups -- no changes to that code path are needed at all.
//   - Checkpoint files (currently only openPMD and Silo support
//     CarpetX::checkpoint_method) instead store CCTK_REAL2 groups
//     *bit-exactly*, as their raw 16-bit storage reinterpreted as
//     `unsigned short`, so that recovery round-trips exactly, including
//     any non-widening-representable bit pattern (e.g. the REAL2 poison
//     value, ipoison_t<CCTK_REAL2> = 0xfead, which is a quiet NaN and
//     would not survive a narrow(widen(x)) round trip bit-for-bit). A
//     temporary amrex::FabArray<amrex::BaseFab<unsigned short>> is built
//     the same way, and likewise handed to the existing generic read/write
//     code path (instantiated for T=unsigned short).
//
// `unsigned short` (rather than std::uint16_t) is used for the raw
// checkpoint payload because it is what openPMD::determineDatatype (->
// Datatype::USHORT), this driver's own db_datatype trait (io_silo.cxx ->
// DB_SHORT), and its mpi_datatype trait (mpi_types.hxx -> MPI_UNSIGNED_
// SHORT) already have ready-made specializations for; the static_assert
// below documents that it must be exactly 16 bits wide for this to be a
// valid bit-for-bit alias of CCTK_REAL2's storage, true on every platform
// this code targets (x86-64 Linux).

#include "driver.hxx"

#include <cctk.h>

#ifdef HAVE_CCTK_REAL2

#include <AMReX_FabArray.H>
#include <AMReX_MultiFab.H>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace CarpetX {

static_assert(sizeof(unsigned short) == sizeof(CCTK_REAL2),
              "unsigned short must be exactly 16 bits wide to alias "
              "CCTK_REAL2's raw storage for checkpointing");

// Raw bit-pattern storage for a CCTK_REAL2 group's checkpoint payload.
using rawMultiFab = amrex::FabArray<amrex::BaseFab<unsigned short> >;

// Allocate a same-shaped (box layout, component count, ghost width)
// destination FabArray as `src`, with unspecified initial contents -- for
// use as a receiving buffer immediately before a read.
template <typename Dst> Dst alloc_like_real2(const hMultiFab &src) {
  return Dst(src.boxArray(), src.DistributionMap(), src.nComp(),
             src.nGrowVect());
}

// Widen `src`'s CCTK_REAL2 values into a same-shaped amrex::fMultiFab, for
// output formats with no fp16 dtype.
inline amrex::fMultiFab widen_real2_to_float(const hMultiFab &src) {
  amrex::fMultiFab dst = alloc_like_real2<amrex::fMultiFab>(src);
  for (amrex::MFIter mfi(src); mfi.isValid(); ++mfi) {
    const amrex::BaseFab<CCTK_REAL2> &sfab = src[mfi];
    amrex::BaseFab<float> &dfab = dst[mfi];
    const std::ptrdiff_t n = std::ptrdiff_t(sfab.box().numPts()) * src.nComp();
    const CCTK_REAL2 *const sp = sfab.dataPtr();
    float *const dp = dfab.dataPtr();
    for (std::ptrdiff_t i = 0; i < n; ++i)
      dp[i] = float(sp[i]);
  }
  return dst;
}

// The mirror of the above, narrowing a float32 amrex::fMultiFab (just read
// from a viz-format file) back down into a CCTK_REAL2 hMultiFab. Not
// exercised by CarpetX's own checkpoint/recovery path (which uses
// rawify_real2/derawify_real2 below instead, for a bit-exact round trip),
// but needed for parity so that CarpetX::filereader_method can also read
// REAL2 initial data from a file another (widening) run produced.
inline void narrow_float_to_real2(const amrex::fMultiFab &src,
                                   hMultiFab &dst) {
  for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi) {
    const amrex::BaseFab<float> &sfab = src[mfi];
    amrex::BaseFab<CCTK_REAL2> &dfab = dst[mfi];
    const std::ptrdiff_t n = std::ptrdiff_t(dfab.box().numPts()) * dst.nComp();
    const float *const sp = sfab.dataPtr();
    CCTK_REAL2 *const dp = dfab.dataPtr();
    for (std::ptrdiff_t i = 0; i < n; ++i)
      dp[i] = CCTK_REAL2(sp[i]);
  }
}

// Reinterpret `src`'s CCTK_REAL2 values as their raw 16-bit storage, for
// checkpointing (D6: checkpoints must round-trip bit-exactly).
inline rawMultiFab rawify_real2(const hMultiFab &src) {
  rawMultiFab dst = alloc_like_real2<rawMultiFab>(src);
  for (amrex::MFIter mfi(src); mfi.isValid(); ++mfi) {
    const amrex::BaseFab<CCTK_REAL2> &sfab = src[mfi];
    amrex::BaseFab<unsigned short> &dfab = dst[mfi];
    const std::ptrdiff_t n = std::ptrdiff_t(sfab.box().numPts()) * src.nComp();
    std::memcpy(dfab.dataPtr(), sfab.dataPtr(), n * sizeof(CCTK_REAL2));
  }
  return dst;
}

// The mirror of the above: reinterpret a raw 16-bit buffer (just read back
// from a checkpoint file) as CCTK_REAL2 values.
inline void derawify_real2(const rawMultiFab &src, hMultiFab &dst) {
  for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi) {
    const amrex::BaseFab<unsigned short> &sfab = src[mfi];
    amrex::BaseFab<CCTK_REAL2> &dfab = dst[mfi];
    const std::ptrdiff_t n = std::ptrdiff_t(dfab.box().numPts()) * dst.nComp();
    std::memcpy(dfab.dataPtr(), sfab.dataPtr(), n * sizeof(CCTK_REAL2));
  }
}

// openPMD/ADIOS2-checkpoint-only workaround
// ------------------------------------------
//
// The installed openPMD-api's ADIOS2 (BP5) backend cannot be used to round
// trip a `rawMultiFab` (Datatype::USHORT / 16-bit) record when more than
// one MPI rank contributes a chunk to it: confirmed empirically (see the
// git history for this comment) that after such a checkpoint is written,
// re-opening it for read finds *zero* of the USHORT-typed meshes in
// `Iteration::meshes` -- `read_iter->meshes.count(meshname)` is false even
// though the record's data and metadata are verifiably present in the
// underlying file (checked directly with `bpls`/`strings`); the analogous
// float32/float64 (CCTK_REAL4/REAL8) meshes in the very same checkpoint,
// written and read by the same multi-rank/multi-chunk code path, are
// unaffected. Single-rank USHORT round trips are unaffected too, so this
// looks like a real, narrow gap in that backend's multi-writer USHORT
// metadata reconstruction, not a bug in CarpetX's own reader/writer code.
//
// The workaround, used only for the openPMD checkpoint transport (Silo's
// own `rawMultiFab` checkpoint round trip above is untouched and unaffected):
// zero-extend the raw 16 bits into the low 16 bits of a 32-bit word and
// bit-pun (via memcpy -- never a numeric conversion, so this is exact for
// every bit pattern, including the REAL2 poison value) that word into an
// amrex::fMultiFab "carrier". That carrier is then handed to openPMD
// exactly like a genuine CCTK_REAL4 group -- the very code path just shown
// to survive this multi-rank round trip -- so no new instantiation of the
// generic read_group/write_group lambdas in io_openpmd.cxx is needed at
// all. The mirror operation on read truncates back down to the low 16
// bits.
inline amrex::fMultiFab widen_raw16_to_carrier(const rawMultiFab &src) {
  amrex::fMultiFab dst(src.boxArray(), src.DistributionMap(), src.nComp(),
                       src.nGrowVect());
  for (amrex::MFIter mfi(src); mfi.isValid(); ++mfi) {
    const amrex::BaseFab<unsigned short> &sfab = src[mfi];
    amrex::BaseFab<float> &dfab = dst[mfi];
    const std::ptrdiff_t n = std::ptrdiff_t(sfab.box().numPts()) * src.nComp();
    const unsigned short *const sp = sfab.dataPtr();
    float *const dp = dfab.dataPtr();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      const std::uint32_t word = sp[i]; // zero-extend, no conversion
      std::memcpy(&dp[i], &word, sizeof dp[i]);
    }
  }
  return dst;
}

inline rawMultiFab narrow_carrier_to_raw16(const amrex::fMultiFab &src) {
  rawMultiFab dst(src.boxArray(), src.DistributionMap(), src.nComp(),
                  src.nGrowVect());
  for (amrex::MFIter mfi(src); mfi.isValid(); ++mfi) {
    const amrex::BaseFab<float> &sfab = src[mfi];
    amrex::BaseFab<unsigned short> &dfab = dst[mfi];
    const std::ptrdiff_t n = std::ptrdiff_t(sfab.box().numPts()) * src.nComp();
    const float *const sp = sfab.dataPtr();
    unsigned short *const dp = dfab.dataPtr();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      std::uint32_t word;
      std::memcpy(&word, &sp[i], sizeof word);
      dp[i] = static_cast<unsigned short>(word & 0xffffu);
    }
  }
  return dst;
}

// Flat-buffer analogues of widen_real2_to_float/narrow_float_to_real2 above,
// for grid scalar/array storage (GHExt::GlobalData::AnyTypeVector), which is
// a raw contiguous buffer of `n` elements rather than an amrex::FabArray.
// Used by the openPMD array/scalar path for CCTK_REAL2 groups' viz output
// (checkpoints instead reinterpret the buffer's raw bits directly as
// `unsigned short`, with no separate widen/narrow step, since that path is
// always single-rank -- see the comment in io_openpmd.cxx).
inline void widen_real2_flat_to_float(const CCTK_REAL2 *const src,
                                      float *const dst,
                                      const std::ptrdiff_t n) {
  for (std::ptrdiff_t i = 0; i < n; ++i)
    dst[i] = float(src[i]);
}
inline void narrow_float_flat_to_real2(const float *const src,
                                       CCTK_REAL2 *const dst,
                                       const std::ptrdiff_t n) {
  for (std::ptrdiff_t i = 0; i < n; ++i)
    dst[i] = CCTK_REAL2(src[i]);
}

} // namespace CarpetX

#endif // #ifdef HAVE_CCTK_REAL2

#endif // #ifndef CARPETX_CARPETX_IO_REAL2_HXX
