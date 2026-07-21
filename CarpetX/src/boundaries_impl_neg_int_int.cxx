#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<NEG, INT, INT>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<NEG, INT, INT>() const;

} // namespace CarpetX
