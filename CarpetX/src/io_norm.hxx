#ifndef CARPETX_CARPETX_IO_NORM_HXX
#define CARPETX_CARPETX_IO_NORM_HXX

#include <cctk.h>

namespace CarpetX {

void OutputNorms(const cGH *restrict cctkGH);

// Per-level norms and fp16 admissibility ratio (out_norm_per_level); see
// io_norm_per_level.cxx
void OutputNormsPerLevel(const cGH *restrict cctkGH);

}

#endif // #define CARPETX_CARPETX_IO_NORM_HXX
