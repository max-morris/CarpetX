#ifndef CARPETX_CARPETX_IO_HXX
#define CARPETX_CARPETX_IO_HXX

#include <cctk.h>

namespace CarpetX {

// Explicit I/O mode for the openPMD/Silo input/output entry points below,
// shared by io.hxx, io_openpmd.hxx and io_silo.hxx: a checkpoint round-trips
// CCTK_REAL2 groups bit-exactly, while a viz-output/filereader-ID call
// widens them to float32 (openPMD/Silo have no fp16 dtype). It does not
// affect CCTK_REAL/REAL8/REAL4 groups either way. See io_real2.hxx. There is
// intentionally no default: every call site must say which it means.
enum class io_mode { viz, checkpoint };

void RecoverGridStructure(cGH *cctkGH);
void RecoverGH(const cGH *cctkGH);
void InputGH(const cGH *cctkGH);

int OutputGH(const cGH *cctkGH);

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_IO_HXX
