#ifndef TESTREAL4_CHECK_EVERY_HXX
#define TESTREAL4_CHECK_EVERY_HXX
#include <cctk.h>
#include <cctk_Parameters.h>
namespace TestReal4 {
// True when a self-check should run this iteration: every `check_every`
// iterations (TestReal4::check_every, shared with TestReal2), and always at
// the final iteration (Cactus::cctk_itlast) so a performance run still ends
// with a verdict.
// Callers must have DECLARE_CCTK_PARAMETERS in scope.
inline bool check_this_iteration(int iteration, CCTK_INT check_every,
                                  CCTK_INT cctk_itlast) {
  return check_every <= 1 || iteration % check_every == 0 ||
         iteration == cctk_itlast;
}
}
#endif
