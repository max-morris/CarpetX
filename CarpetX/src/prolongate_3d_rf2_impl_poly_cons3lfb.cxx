#include "prolongate_3d_rf2_impl.hxx"

namespace CarpetX {

template <typename T>
const std::map<int, std::array<InterpolaterT<T> *, 8> > &prolongate_poly_cons3lfb_3d_rf2_table() {
  // Interpolate polynomially in vertex centred directions and conserve
  // with 3rd order accuracy and a linear fallback in cell centred
  // directions

  static prolongate_3d_rf2<VC, VC, VC, POLY, POLY, POLY, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c000_o1;
  static prolongate_3d_rf2<VC, VC, CC, POLY, POLY, CONS, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c001_o1;
  static prolongate_3d_rf2<VC, CC, VC, POLY, CONS, POLY, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c010_o1;
  static prolongate_3d_rf2<VC, CC, CC, POLY, CONS, CONS, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c011_o1;
  static prolongate_3d_rf2<CC, VC, VC, CONS, POLY, POLY, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c100_o1;
  static prolongate_3d_rf2<CC, VC, CC, CONS, POLY, CONS, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c101_o1;
  static prolongate_3d_rf2<CC, CC, VC, CONS, CONS, POLY, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c110_o1;
  static prolongate_3d_rf2<CC, CC, CC, CONS, CONS, CONS, 1, 1, 1, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c111_o1;

  static prolongate_3d_rf2<VC, VC, VC, POLY, POLY, POLY, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c000_o3;
  static prolongate_3d_rf2<VC, VC, CC, POLY, POLY, CONS, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c001_o3;
  static prolongate_3d_rf2<VC, CC, VC, POLY, CONS, POLY, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c010_o3;
  static prolongate_3d_rf2<VC, CC, CC, POLY, CONS, CONS, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c011_o3;
  static prolongate_3d_rf2<CC, VC, VC, CONS, POLY, POLY, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c100_o3;
  static prolongate_3d_rf2<CC, VC, CC, CONS, POLY, CONS, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c101_o3;
  static prolongate_3d_rf2<CC, CC, VC, CONS, CONS, POLY, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c110_o3;
  static prolongate_3d_rf2<CC, CC, CC, CONS, CONS, CONS, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c111_o3;

  static prolongate_3d_rf2<VC, VC, VC, POLY, POLY, POLY, 5, 5, 5, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c000_o5;
  static prolongate_3d_rf2<VC, VC, CC, POLY, POLY, CONS, 5, 5, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c001_o5;
  static prolongate_3d_rf2<VC, CC, VC, POLY, CONS, POLY, 5, 3, 5, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c010_o5;
  static prolongate_3d_rf2<VC, CC, CC, POLY, CONS, CONS, 5, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c011_o5;
  static prolongate_3d_rf2<CC, VC, VC, CONS, POLY, POLY, 3, 5, 5, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c100_o5;
  static prolongate_3d_rf2<CC, VC, CC, CONS, POLY, CONS, 3, 5, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c101_o5;
  static prolongate_3d_rf2<CC, CC, VC, CONS, CONS, POLY, 3, 3, 5, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c110_o5;
  static prolongate_3d_rf2<CC, CC, CC, CONS, CONS, CONS, 3, 3, 3, FB_LINEAR, T>
      prolongate_poly_cons3lfb_3d_rf2_c111_o5;

  #if 0
  static prolongate_3d_rf2<VC, VC, VC, POLY, POLY, POLY, 7, 7, 7, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c000_o7;
  static prolongate_3d_rf2<VC, VC, CC, POLY, POLY, CONS, 7, 7, 3, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c001_o7;
  static prolongate_3d_rf2<VC, CC, VC, POLY, CONS, POLY, 7, 3, 7, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c010_o7;
  static prolongate_3d_rf2<VC, CC, CC, POLY, CONS, CONS, 7, 3, 3, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c011_o7;
  static prolongate_3d_rf2<CC, VC, VC, CONS, POLY, POLY, 3, 7, 7, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c100_o7;
  static prolongate_3d_rf2<CC, VC, CC, CONS, POLY, CONS, 3, 7, 3, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c101_o7;
  static prolongate_3d_rf2<CC, CC, VC, CONS, CONS, POLY, 3, 3, 7, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c110_o7;
  static prolongate_3d_rf2<CC, CC, CC, CONS, CONS, CONS, 3, 3, 3, FB_LINEAR, T>    prolongate_poly_cons3lfb_3d_rf2_c111_o7;
  #endif

  static const std::map<int, std::array<InterpolaterT<T> *, 8> > table{
          {1,
           {
               &prolongate_poly_cons3lfb_3d_rf2_c000_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c001_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c010_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c011_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c100_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c101_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c110_o1,
               &prolongate_poly_cons3lfb_3d_rf2_c111_o1,
           }},
          {3,
           {
               &prolongate_poly_cons3lfb_3d_rf2_c000_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c001_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c010_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c011_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c100_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c101_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c110_o3,
               &prolongate_poly_cons3lfb_3d_rf2_c111_o3,
           }},
          {5,
           {
               &prolongate_poly_cons3lfb_3d_rf2_c000_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c001_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c010_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c011_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c100_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c101_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c110_o5,
               &prolongate_poly_cons3lfb_3d_rf2_c111_o5,
           }},
  #if 0
          {7,
           {
               &prolongate_poly_cons3lfb_3d_rf2_c000_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c001_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c010_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c011_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c100_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c101_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c110_o7,
               &prolongate_poly_cons3lfb_3d_rf2_c111_o7,
           }},
  #endif
      };

  return table;
}

template const std::map<int, std::array<InterpolaterT<CCTK_REAL> *, 8> > &
prolongate_poly_cons3lfb_3d_rf2_table<CCTK_REAL>();
template const std::map<int, std::array<InterpolaterT<CCTK_REAL4> *, 8> > &
prolongate_poly_cons3lfb_3d_rf2_table<CCTK_REAL4>();

} // namespace CarpetX
