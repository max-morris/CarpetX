#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<INT, INT, NEG>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<INT, INT, NEG>() const;

} // namespace CarpetX
