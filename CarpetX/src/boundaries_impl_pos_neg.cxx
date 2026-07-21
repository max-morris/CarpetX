#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<NEG, POS, NEG>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<NEG, POS, NEG>() const;
template void BoundaryCondition<CCTK_REAL>::apply_on_face<INT, POS, NEG>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<INT, POS, NEG>() const;
template void BoundaryCondition<CCTK_REAL>::apply_on_face<POS, POS, NEG>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<POS, POS, NEG>() const;

} // namespace CarpetX
