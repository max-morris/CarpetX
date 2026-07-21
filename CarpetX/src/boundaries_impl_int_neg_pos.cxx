#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<INT, NEG, POS>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<INT, NEG, POS>() const;

} // namespace CarpetX
