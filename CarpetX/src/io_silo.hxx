#ifndef CARPETX_CARPETX_IO_SILO_HXX
#define CARPETX_CARPETX_IO_SILO_HXX

#include <cctk.h>

#include "io.hxx"

#ifdef HAVE_CAPABILITY_Silo

#include <string>
#include <vector>

namespace CarpetX {

int InputSiloParameters(const std::string &input_dir,
                        const std::string &input_file);
void InputSiloGridStructure(cGH *cctkGH, const std::string &input_dir,
                            const std::string &input_file, int input_iteration);
// `mode`: see the identical parameter on io_openpmd.hxx's
// InputOpenPMD/OutputOpenPMD -- same D6 rationale (bit-exact raw CCTK_REAL2
// checkpoint payload vs. widened-to-float32 visualization output). No
// default: every call site must say which it means.
void InputSilo(const cGH *cctkGH, const std::vector<bool> &input_group,
               const std::string &input_dir, const std::string &input_file,
               io_mode mode);

void OutputSilo(const cGH *cctkGH, const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                io_mode mode);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_SILO_HXX
