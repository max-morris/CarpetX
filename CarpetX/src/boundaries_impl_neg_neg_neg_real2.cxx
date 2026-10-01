#include "boundaries_impl.hxx"

namespace CarpetX {

#ifdef HAVE_CCTK_REAL2
template void
BoundaryCondition<CCTK_REAL2>::apply_on_face<NEG, NEG, NEG>() const;
#endif

} // namespace CarpetX
