#ifndef CARPETX_CARPETX_REAL2_LIMITS_HXX
#define CARPETX_CARPETX_REAL2_LIMITS_HXX

// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc): a small kit of
// portable replacements for std::numeric_limits<T>::foo() call sites in
// generic (T-templated) code that may be instantiated with T=CCTK_REAL2.
//
// Under gcc, CCTK_REAL2 is `_Float16` (see cctk_Types.h), for which GCC's
// own <limits> header supplies a correct std::numeric_limits
// specialization. Under nvcc, CCTK_REAL2 is `__half`, for which no
// std::numeric_limits specialization exists in the CUDA toolkit headers:
// the *unspecialized* primary std::numeric_limits<T> template is still
// well-formed for any T (so this is not a compile error), but its members
// (epsilon(), infinity(), etc.) silently return meaningless defaults (0,
// for a floating-point-like type) instead of the real binary16 values --
// a silent correctness bug, not a build failure.
//
// We deliberately do not fix this by specializing std::numeric_limits for
// __half ourselves: that would mean adding a specialization of a standard
// library template for a vendor (CUDA) type, in a shared header seen by
// both gcc and nvcc translation units, which risks clashing with any
// specialization a future CUDA toolkit version supplies itself. Instead,
// the helpers below derive each binary16 constant directly from its known
// bit pattern / mathematical definition, so they are correct regardless of
// which underlying type CCTK_REAL2 is mapped to (_Float16 or __half) --
// and, for T other than CCTK_REAL2, they are a pure passthrough to
// std::numeric_limits<T>, so this is a no-op for CCTK_REAL8/CCTK_REAL4.

#include "loop.hxx" // for CCTK_DEVICE/CCTK_HOST

#include <cctk.h>

#include <limits>
#include <type_traits>

namespace CarpetX {

// binary16 (IEEE 754 half precision) machine epsilon: 10 mantissa bits =>
// 2^-10 = 0.0009765625, itself an exact power of two, hence exactly
// representable in binary16 (so the conversion below is exact on both
// _Float16 and __half).
template <typename T>
CCTK_DEVICE CCTK_HOST constexpr T portable_epsilon() {
#ifdef HAVE_CCTK_REAL2
  if constexpr (std::is_same_v<T, CCTK_REAL2>) {
    return T(0.0009765625);
  } else
#endif
  {
    return std::numeric_limits<T>::epsilon();
  }
}

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_REAL2_LIMITS_HXX
