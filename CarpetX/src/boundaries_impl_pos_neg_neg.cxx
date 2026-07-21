#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<POS, NEG, NEG>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<POS, NEG, NEG>() const;

} // namespace CarpetX
