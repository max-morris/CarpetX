#include "prolongate_3d_rf2_impl.hxx"

namespace CarpetX {

template <typename T>
const std::map<int, std::array<InterpolaterT<T> *, 8> > &prolongate_cons_3d_rf2_table() {
  // Conservative interpolation

  static prolongate_3d_rf2<VC, VC, VC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c000_o0;
  static prolongate_3d_rf2<VC, VC, CC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c001_o0;
  static prolongate_3d_rf2<VC, CC, VC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c010_o0;
  static prolongate_3d_rf2<VC, CC, CC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c011_o0;
  static prolongate_3d_rf2<CC, VC, VC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c100_o0;
  static prolongate_3d_rf2<CC, VC, CC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c101_o0;
  static prolongate_3d_rf2<CC, CC, VC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c110_o0;
  static prolongate_3d_rf2<CC, CC, CC, CONS, CONS, CONS, 0, 0, 0, FB_NONE, T>
      prolongate_cons_3d_rf2_c111_o0;

  static prolongate_3d_rf2<VC, VC, VC, CONS, CONS, CONS, 1, 1, 1, FB_NONE, T>
      prolongate_cons_3d_rf2_c000_o1;
  static prolongate_3d_rf2<VC, VC, CC, CONS, CONS, CONS, 1, 1, 2, FB_NONE, T>
      prolongate_cons_3d_rf2_c001_o1;
  static prolongate_3d_rf2<VC, CC, VC, CONS, CONS, CONS, 1, 2, 1, FB_NONE, T>
      prolongate_cons_3d_rf2_c010_o1;
  static prolongate_3d_rf2<VC, CC, CC, CONS, CONS, CONS, 1, 2, 2, FB_NONE, T>
      prolongate_cons_3d_rf2_c011_o1;
  static prolongate_3d_rf2<CC, VC, VC, CONS, CONS, CONS, 2, 1, 1, FB_NONE, T>
      prolongate_cons_3d_rf2_c100_o1;
  static prolongate_3d_rf2<CC, VC, CC, CONS, CONS, CONS, 2, 1, 2, FB_NONE, T>
      prolongate_cons_3d_rf2_c101_o1;
  static prolongate_3d_rf2<CC, CC, VC, CONS, CONS, CONS, 2, 2, 1, FB_NONE, T>
      prolongate_cons_3d_rf2_c110_o1;
  static prolongate_3d_rf2<CC, CC, CC, CONS, CONS, CONS, 2, 2, 2, FB_NONE, T>
      prolongate_cons_3d_rf2_c111_o1;

  static const std::map<int, std::array<InterpolaterT<T> *, 8> > table{
          {0,
           {
               &prolongate_cons_3d_rf2_c000_o0,
               &prolongate_cons_3d_rf2_c001_o0,
               &prolongate_cons_3d_rf2_c010_o0,
               &prolongate_cons_3d_rf2_c011_o0,
               &prolongate_cons_3d_rf2_c100_o0,
               &prolongate_cons_3d_rf2_c101_o0,
               &prolongate_cons_3d_rf2_c110_o0,
               &prolongate_cons_3d_rf2_c111_o0,
           }},
          {1,
           {
               &prolongate_cons_3d_rf2_c000_o1,
               &prolongate_cons_3d_rf2_c001_o1,
               &prolongate_cons_3d_rf2_c010_o1,
               &prolongate_cons_3d_rf2_c011_o1,
               &prolongate_cons_3d_rf2_c100_o1,
               &prolongate_cons_3d_rf2_c101_o1,
               &prolongate_cons_3d_rf2_c110_o1,
               &prolongate_cons_3d_rf2_c111_o1,
           }},
      };

  return table;
}

template const std::map<int, std::array<InterpolaterT<CCTK_REAL> *, 8> > &
prolongate_cons_3d_rf2_table<CCTK_REAL>();
template const std::map<int, std::array<InterpolaterT<CCTK_REAL4> *, 8> > &
prolongate_cons_3d_rf2_table<CCTK_REAL4>();

} // namespace CarpetX
