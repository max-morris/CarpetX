#include "prolongate_3d_rf2_impl.hxx"

namespace CarpetX {

template <typename T>
const std::map<int, std::array<InterpolaterT<T> *, 8> > &prolongate_eno_3d_rf2_table() {
  // ENO (tensor product) interpolation

  static prolongate_3d_rf2<VC, VC, VC, POLY, POLY, POLY, 3, 3, 3, FB_NONE, T>
      prolongate_eno_3d_rf2_c000_o3;
  static prolongate_3d_rf2<VC, VC, CC, POLY, POLY, ENO, 3, 3, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c001_o3;
  static prolongate_3d_rf2<VC, CC, VC, POLY, ENO, POLY, 3, 2, 3, FB_NONE, T>
      prolongate_eno_3d_rf2_c010_o3;
  static prolongate_3d_rf2<VC, CC, CC, POLY, ENO, ENO, 3, 2, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c011_o3;
  static prolongate_3d_rf2<CC, VC, VC, ENO, POLY, POLY, 2, 3, 3, FB_NONE, T>
      prolongate_eno_3d_rf2_c100_o3;
  static prolongate_3d_rf2<CC, VC, CC, ENO, POLY, ENO, 2, 3, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c101_o3;
  static prolongate_3d_rf2<CC, CC, VC, ENO, ENO, POLY, 2, 2, 3, FB_NONE, T>
      prolongate_eno_3d_rf2_c110_o3;
  static prolongate_3d_rf2<CC, CC, CC, ENO, ENO, ENO, 2, 2, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c111_o3;

  static prolongate_3d_rf2<VC, VC, VC, POLY, POLY, POLY, 5, 5, 5, FB_NONE, T>
      prolongate_eno_3d_rf2_c000_o5;
  static prolongate_3d_rf2<VC, VC, CC, POLY, POLY, ENO, 5, 5, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c001_o5;
  static prolongate_3d_rf2<VC, CC, VC, POLY, ENO, POLY, 5, 2, 5, FB_NONE, T>
      prolongate_eno_3d_rf2_c010_o5;
  static prolongate_3d_rf2<VC, CC, CC, POLY, ENO, ENO, 5, 2, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c011_o5;
  static prolongate_3d_rf2<CC, VC, VC, ENO, POLY, POLY, 2, 5, 5, FB_NONE, T>
      prolongate_eno_3d_rf2_c100_o5;
  static prolongate_3d_rf2<CC, VC, CC, ENO, POLY, ENO, 2, 5, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c101_o5;
  static prolongate_3d_rf2<CC, CC, VC, ENO, ENO, POLY, 2, 2, 5, FB_NONE, T>
      prolongate_eno_3d_rf2_c110_o5;
  static prolongate_3d_rf2<CC, CC, CC, ENO, ENO, ENO, 2, 2, 2, FB_NONE, T>
      prolongate_eno_3d_rf2_c111_o5;

  static const std::map<int, std::array<InterpolaterT<T> *, 8> > table{
          {3,
           {
               &prolongate_eno_3d_rf2_c000_o3,
               &prolongate_eno_3d_rf2_c001_o3,
               &prolongate_eno_3d_rf2_c010_o3,
               &prolongate_eno_3d_rf2_c011_o3,
               &prolongate_eno_3d_rf2_c100_o3,
               &prolongate_eno_3d_rf2_c101_o3,
               &prolongate_eno_3d_rf2_c110_o3,
               &prolongate_eno_3d_rf2_c111_o3,
           }},
          {5,
           {
               &prolongate_eno_3d_rf2_c000_o5,
               &prolongate_eno_3d_rf2_c001_o5,
               &prolongate_eno_3d_rf2_c010_o5,
               &prolongate_eno_3d_rf2_c011_o5,
               &prolongate_eno_3d_rf2_c100_o5,
               &prolongate_eno_3d_rf2_c101_o5,
               &prolongate_eno_3d_rf2_c110_o5,
               &prolongate_eno_3d_rf2_c111_o5,
           }},
      };

  return table;
}

template const std::map<int, std::array<InterpolaterT<CCTK_REAL> *, 8> > &
prolongate_eno_3d_rf2_table<CCTK_REAL>();
template const std::map<int, std::array<InterpolaterT<CCTK_REAL4> *, 8> > &
prolongate_eno_3d_rf2_table<CCTK_REAL4>();
#ifdef HAVE_CCTK_REAL2
template const std::map<int, std::array<InterpolaterT<CCTK_REAL2> *, 8> > &
prolongate_eno_3d_rf2_table<CCTK_REAL2>();
#endif

} // namespace CarpetX
