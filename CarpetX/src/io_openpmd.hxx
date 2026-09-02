#ifndef CARPETX_CARPETX_IO_OPENPMD_HXX
#define CARPETX_CARPETX_IO_OPENPMD_HXX

#include <cctk.h>

#include "io.hxx"

#ifdef HAVE_CAPABILITY_openPMD_api

#include <string>
#include <vector>

namespace CarpetX {

int InputOpenPMDParameters(const std::string &input_dir,
                           const std::string &input_file);
void InputOpenPMDGridStructure(cGH *cctkGH, const std::string &input_dir,
                               const std::string &input_file,
                               int input_iteration);
// `mode` distinguishes a checkpoint/recovery call from a regular
// visualization-output/filereader-ID call: per D6, CCTK_REAL2 groups are
// stored bit-exactly (raw 16-bit payload) when checkpointing, but widened
// to float32 for visualization output (which has no fp16 dtype). It does
// not affect CCTK_REAL/REAL8/REAL4 groups either way. See io_real2.hxx.
// There is no default: passing the wrong mode silently corrupts REAL2
// groups (see the `carpetx_real2_encoding` attribute check in
// io_openpmd.cxx), so every call site must say which it means.
void InputOpenPMD(const cGH *cctkGH, const std::vector<bool> &input_group,
                  const std::string &input_dir, const std::string &input_file,
                  io_mode mode);

void OutputOpenPMD(const cGH *cctkGH, const std::vector<bool> &output_group,
                   const std::string &output_dir,
                   const std::string &output_file, io_mode mode);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_OPENPMD_HXX
