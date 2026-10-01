#include "prolongate_3d_rf2_impl_poly_cons3lfb.hxx"

namespace CarpetX {

// Instantiate the operators for each precision in a separate file to keep
// compile times short.
template const std::map<int, std::array<InterpolaterT<CCTK_REAL> *, 8> > &
prolongate_poly_cons3lfb_3d_rf2_table<CCTK_REAL>();

} // namespace CarpetX
