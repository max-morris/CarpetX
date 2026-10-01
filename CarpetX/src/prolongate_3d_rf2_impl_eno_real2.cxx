#include "prolongate_3d_rf2_impl_eno.hxx"

namespace CarpetX {

// Instantiate the operators for each precision in a separate file to keep
// compile times short.
#ifdef HAVE_CCTK_REAL2
template const std::map<int, std::array<InterpolaterT<CCTK_REAL2> *, 8> > &
prolongate_eno_3d_rf2_table<CCTK_REAL2>();
#endif

} // namespace CarpetX
