#include "prolongate_3d_rf2_impl_cons.hxx"

namespace CarpetX {

// Instantiate the operators for each precision in a separate file to keep
// compile times short.
template const std::map<int, std::array<InterpolaterT<CCTK_REAL> *, 8> > &
prolongate_cons_3d_rf2_table<CCTK_REAL>();

} // namespace CarpetX
