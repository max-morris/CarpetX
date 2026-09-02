#ifndef CARPETX_CARPETX_REAL2_LIMITS_HXX
#define CARPETX_CARPETX_REAL2_LIMITS_HXX

// H2 (mixed precision, CCTK_REAL2 -> __half under nvcc; -> _Float16 under
// gcc/g++): a small kit of portable replacements for
// std::numeric_limits<T>::foo() call sites in generic (T-templated) code
// that may be instantiated with T=CCTK_REAL2.
//
// Under nvcc, CCTK_REAL2 is `__half`, for which no std::numeric_limits
// specialization exists in the CUDA toolkit headers: the *unspecialized*
// primary std::numeric_limits<T> template is still well-formed for any T
// (so this is not a compile error), but its members (epsilon(),
// infinity(), etc.) silently return meaningless defaults (0, for a
// floating-point-like type) instead of the real binary16 values -- a
// silent correctness bug, not a build failure.
//
// Under gcc/g++, CCTK_REAL2 is `_Float16` (see cctk_Types.h). GCC's own
// libstdc++ <limits> header only specializes std::numeric_limits for the
// extended floating-point types (including _Float16) when compiling in
// C++23 mode (`__STDCPP_FLOAT16_T__`); this project builds with
// `-std=c++17` (see the option list), under which
// std::numeric_limits<_Float16> hits the very same unspecialized primary
// template as __half does under nvcc, with the very same silent-zero
// failure mode. (Measured with the project's own compiler: under
// -std=c++17/-std=c++20, `numeric_limits<_Float16>::is_specialized` is 0
// and `epsilon()`/`max()` return 0; only -std=c++23 gives the "obviously
// correct" specialization one might expect from a native compiler type.)
// So this header's helpers are required on the host build too, not only
// under nvcc.
//
// We deliberately do not fix this by specializing std::numeric_limits for
// __half/_Float16 ourselves: that would mean adding a specialization of a
// standard library template for a type this project does not own (a CUDA
// vendor type, or -- for _Float16 -- a type libstdc++ itself may
// specialize once this project moves to C++23), in a shared header seen by
// both gcc and nvcc translation units, which risks clashing with any
// specialization a future toolchain version supplies itself. Instead, the
// helpers below derive each binary16 constant directly from its known bit
// pattern / mathematical definition, so they are correct regardless of
// which underlying type CCTK_REAL2 is mapped to (_Float16 or __half) and
// regardless of the C++ standard version -- and, for T other than
// CCTK_REAL2, they are a pure passthrough to std::numeric_limits<T>, so
// this is a no-op for CCTK_REAL8/CCTK_REAL4.
//
// Convention: generic (T-templated) code that might ever be instantiated
// with T=CCTK_REAL2 should use these helpers instead of bare
// std::numeric_limits<T>::foo() -- the latter silently returns 0 for
// CCTK_REAL2 with no diagnostic, on every toolchain this project supports.

#include "loop.hxx" // for CCTK_DEVICE/CCTK_HOST

#include <cctk.h>

#include <cstdint>
#include <cstring>
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

// binary16 largest finite value: 65504 = (2 - 2^-10) * 2^15, exactly
// representable in binary16 (so the conversion below is exact).
template <typename T>
CCTK_DEVICE CCTK_HOST constexpr T portable_max() {
#ifdef HAVE_CCTK_REAL2
  if constexpr (std::is_same_v<T, CCTK_REAL2>) {
    return T(65504.0);
  } else
#endif
  {
    return std::numeric_limits<T>::max();
  }
}

// binary16 smallest positive normalized value: 2^-14, an exact power of
// two, hence exactly representable in binary16.
template <typename T>
CCTK_DEVICE CCTK_HOST constexpr T portable_min() {
#ifdef HAVE_CCTK_REAL2
  if constexpr (std::is_same_v<T, CCTK_REAL2>) {
    return T(6.103515625e-05);
  } else
#endif
  {
    return std::numeric_limits<T>::min();
  }
}

// binary16 number of mantissa bits including the implicit leading bit
// (matches std::numeric_limits<T>::digits' definition: 10 explicit mantissa
// bits + 1 implicit leading bit = 11).
template <typename T>
CCTK_DEVICE CCTK_HOST constexpr int portable_digits() {
#ifdef HAVE_CCTK_REAL2
  if constexpr (std::is_same_v<T, CCTK_REAL2>) {
    return 11;
  } else
#endif
  {
    return std::numeric_limits<T>::digits;
  }
}

// binary16 +infinity: sign=0, exponent=0x1f (all ones), mantissa=0 -> bit
// pattern 0x7c00. Built directly from the bit pattern (a 2-byte memcpy,
// device-capable -- see io_real2.hxx's rawify_real2 for the same idiom)
// rather than from a double/float->T conversion, so this does not depend
// on the compiler correctly rounding an out-of-range conversion to
// infinity on every toolchain.
template <typename T> CCTK_DEVICE CCTK_HOST inline T portable_infinity() {
#ifdef HAVE_CCTK_REAL2
  if constexpr (std::is_same_v<T, CCTK_REAL2>) {
    constexpr std::uint16_t ibits = 0x7c00;
    T result;
    std::memcpy(&result, &ibits, sizeof result);
    return result;
  } else
#endif
  {
    return std::numeric_limits<T>::infinity();
  }
}

// binary16 quiet NaN: sign=0, exponent=0x1f (all ones), mantissa=0x200
// (nonzero, top bit set so this is quiet, not signalling) -> bit pattern
// 0x7e00. Built directly from the bit pattern, as with portable_infinity
// above.
template <typename T> CCTK_DEVICE CCTK_HOST inline T portable_quiet_NaN() {
#ifdef HAVE_CCTK_REAL2
  if constexpr (std::is_same_v<T, CCTK_REAL2>) {
    constexpr std::uint16_t ibits = 0x7e00;
    T result;
    std::memcpy(&result, &ibits, sizeof result);
    return result;
  } else
#endif
  {
    return std::numeric_limits<T>::quiet_NaN();
  }
}

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_REAL2_LIMITS_HXX
