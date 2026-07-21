#ifndef CARPETX_CARPETX_MPI_TYPEMAP_REAL2_HXX
#define CARPETX_CARPETX_MPI_TYPEMAP_REAL2_HXX

// amrex::ParallelDescriptor::Mpi_typemap<CCTK_REAL2> specialization.
//
// Background: AMReX's own FillBoundary/ParallelCopy communication path
// (AMReX_FabArrayCommI.H's PostSnds/PostRcvs) packs FAB data into raw `char`
// buffers and sends/receives them via ParallelDescriptor::Asend<char>/
// Arecv<char> (i.e. through Mpi_typemap<char>::type(), never through
// Mpi_typemap<value_type>::type() for the FAB's actual element type) -- this
// was verified by reading AMReX_FBI.H/AMReX_FabArrayCommI.H, which pass
// `char*`/byte counts throughout, not typed buffers. So this specialization
// is not on the hot path of CarpetX's day-to-day ghost-fill communication
// for hMultiFab (FabArray<BaseFab<CCTK_REAL2>>) groups.
//
// It is still provided, per the design's D3 decision, because a handful of
// other AMReX code paths (e.g. AMReX_ParallelReduce.H's
// ParallelAllReduce/ParallelReduce helpers) look up
// ParallelDescriptor::Mpi_typemap<T>::type() directly for an arbitrary
// element type T. The primary template
// (amrex::ParallelDescriptor::Mpi_typemap<T>::type(), declared in
// AMReX_ccse-mpi.H) is declared but never *defined* generically -- only
// specific types (char, short, int, ..., float, double, lull_t) have a
// definition supplied by AMReX's own compiled library -- so any such call
// for T=CCTK_REAL2 would otherwise compile fine (the declaration is visible)
// but fail to *link*. This header closes that gap pre-emptively.
//
// There is no standard MPI datatype for IEEE binary16 (MPI predates fp16),
// so this constructs a 2-byte opaque type via MPI_Type_contiguous(2,
// MPI_BYTE) instead, exactly like CarpetX's own reduction_mpi_datatype<T,D>()
// (reduction.cxx) builds ad-hoc contiguous MPI types for its reduction<T,D>
// structs. This is safe because Mpi_typemap<CCTK_REAL2> only ever needs to
// describe the raw storage/transfer of CCTK_REAL2 values with the right
// size and alignment -- never as an MPI reduction operand (reductions of
// REAL2 grid functions widen to double and reduce as double, per D7), so it
// does not need to match a real MPI floating-point datatype.
//
// Construction is deferred to first use (rather than done eagerly at
// namespace/static scope) because it requires MPI_Init to already have run;
// it is memoized in a function-local static, whose initialization the
// C++11-and-later standard guarantees is both performed at most once and
// thread-safe even under concurrent first calls ("magic statics") -- at
// least as thread-safe as AMReX's own analogous
// ParallelDescriptor::Mpi_typemap<ValLocPair<TV,TI>>::type()
// (AMReX_ParallelDescriptor.H), which memoizes its own ad-hoc MPI datatype
// via an explicit "if (mpi_type == MPI_DATATYPE_NULL)" null-check instead of
// a function-local static (presumably for portability to pre-C++11
// compilers AMReX once supported); the two approaches are equivalent for a
// single-threaded first call, but the function-local static additionally
// guarantees correctness if the first call itself happens concurrently from
// multiple threads.

#include <cctk.h>

#if defined HAVE_CCTK_REAL2 && defined AMREX_USE_MPI

#include <AMReX_ccse-mpi.H>

namespace amrex {
namespace ParallelDescriptor {

template <> struct Mpi_typemap<CCTK_REAL2> {
  static MPI_Datatype type() {
    static const MPI_Datatype datatype = [] {
      MPI_Datatype dt;
      MPI_Type_contiguous(2, MPI_BYTE, &dt);
      MPI_Type_commit(&dt);
      return dt;
    }();
    return datatype;
  }
};

} // namespace ParallelDescriptor
} // namespace amrex

#endif // #if defined HAVE_CCTK_REAL2 && defined AMREX_USE_MPI

#endif // #ifndef CARPETX_CARPETX_MPI_TYPEMAP_REAL2_HXX
