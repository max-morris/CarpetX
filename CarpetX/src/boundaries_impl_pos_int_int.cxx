#include "boundaries_impl.hxx"

namespace CarpetX {

template void BoundaryCondition<CCTK_REAL>::apply_on_face<POS, INT, INT>() const;
template void BoundaryCondition<CCTK_REAL4>::apply_on_face<POS, INT, INT>() const;
#ifdef HAVE_CCTK_REAL2
template void BoundaryCondition<CCTK_REAL2>::apply_on_face<POS, INT, INT>() const;
#endif

} // namespace CarpetX
