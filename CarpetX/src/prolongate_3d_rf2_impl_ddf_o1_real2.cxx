#include "prolongate_3d_rf2_impl_ddf.hxx"

namespace CarpetX {

#ifdef HAVE_CCTK_REAL2
template std::array<InterpolaterT<CCTK_REAL2> *, 8>
prolongate_ddf_3d_rf2_o1<CCTK_REAL2>();
#endif

} // namespace CarpetX
