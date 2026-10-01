#include "prolongate_3d_rf2_impl_ddf.hxx"

namespace CarpetX {

// The operators for each order and precision are instantiated in separate
// files (prolongate_3d_rf2_impl_ddf_o*_real*.cxx) to keep compile times short.
extern template std::array<InterpolaterT<CCTK_REAL> *, 8>
prolongate_ddf_3d_rf2_o1<CCTK_REAL>();
extern template std::array<InterpolaterT<CCTK_REAL4> *, 8>
prolongate_ddf_3d_rf2_o1<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
extern template std::array<InterpolaterT<CCTK_REAL2> *, 8>
prolongate_ddf_3d_rf2_o1<CCTK_REAL2>();
#endif
extern template std::array<InterpolaterT<CCTK_REAL> *, 8>
prolongate_ddf_3d_rf2_o3<CCTK_REAL>();
extern template std::array<InterpolaterT<CCTK_REAL4> *, 8>
prolongate_ddf_3d_rf2_o3<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
extern template std::array<InterpolaterT<CCTK_REAL2> *, 8>
prolongate_ddf_3d_rf2_o3<CCTK_REAL2>();
#endif
extern template std::array<InterpolaterT<CCTK_REAL> *, 8>
prolongate_ddf_3d_rf2_o5<CCTK_REAL>();
extern template std::array<InterpolaterT<CCTK_REAL4> *, 8>
prolongate_ddf_3d_rf2_o5<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
extern template std::array<InterpolaterT<CCTK_REAL2> *, 8>
prolongate_ddf_3d_rf2_o5<CCTK_REAL2>();
#endif
extern template std::array<InterpolaterT<CCTK_REAL> *, 8>
prolongate_ddf_3d_rf2_o7<CCTK_REAL>();
extern template std::array<InterpolaterT<CCTK_REAL4> *, 8>
prolongate_ddf_3d_rf2_o7<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
extern template std::array<InterpolaterT<CCTK_REAL2> *, 8>
prolongate_ddf_3d_rf2_o7<CCTK_REAL2>();
#endif

template <typename T>
const std::map<int, std::array<InterpolaterT<T> *, 8> > &
prolongate_ddf_3d_rf2_table() {
  static const std::map<int, std::array<InterpolaterT<T> *, 8> > table{
      {1, prolongate_ddf_3d_rf2_o1<T>()},
      {3, prolongate_ddf_3d_rf2_o3<T>()},
      {5, prolongate_ddf_3d_rf2_o5<T>()},
      {7, prolongate_ddf_3d_rf2_o7<T>()}};
  return table;
}

template const std::map<int, std::array<InterpolaterT<CCTK_REAL> *, 8> > &
prolongate_ddf_3d_rf2_table<CCTK_REAL>();
template const std::map<int, std::array<InterpolaterT<CCTK_REAL4> *, 8> > &
prolongate_ddf_3d_rf2_table<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
template const std::map<int, std::array<InterpolaterT<CCTK_REAL2> *, 8> > &
prolongate_ddf_3d_rf2_table<CCTK_REAL2>();
#endif

} // namespace CarpetX
