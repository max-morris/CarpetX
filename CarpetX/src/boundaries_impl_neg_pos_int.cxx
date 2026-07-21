#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<NEG, POS, INT>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<NEG, POS, INT>() const;

} // namespace CarpetX
