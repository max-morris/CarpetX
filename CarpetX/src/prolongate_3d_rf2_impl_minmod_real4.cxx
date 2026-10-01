#include "prolongate_3d_rf2_impl_minmod.hxx"

namespace CarpetX {

// Instantiate the operators for each precision in a separate file to keep
// compile times short.
template const std::map<int, std::array<InterpolaterT<CCTK_REAL4> *, 8> > &
prolongate_minmod_3d_rf2_table<CCTK_REAL4>();

} // namespace CarpetX
