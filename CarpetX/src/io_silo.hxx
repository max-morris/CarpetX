#ifndef CARPETX_CARPETX_IO_SILO_HXX
#define CARPETX_CARPETX_IO_SILO_HXX

#include <cctk.h>

#ifdef HAVE_CAPABILITY_Silo

#include <string>
#include <vector>

namespace CarpetX {

int InputSiloParameters(const std::string &input_dir,
                        const std::string &input_file);
void InputSiloGridStructure(cGH *cctkGH, const std::string &input_dir,
                            const std::string &input_file, int input_iteration);
// `is_checkpoint`: see the identical parameter on io_openpmd.hxx's
// InputOpenPMD/OutputOpenPMD -- same D6 rationale (bit-exact raw CCTK_REAL2
// checkpoint payload vs. widened-to-float32 visualization output).
void InputSilo(const cGH *cctkGH, const std::vector<bool> &input_group,
               const std::string &input_dir, const std::string &input_file,
               bool is_checkpoint = false);

void OutputSilo(const cGH *cctkGH, const std::vector<bool> &output_group,
                const std::string &output_dir, const std::string &output_file,
                bool is_checkpoint = false);

} // namespace CarpetX

#endif

#endif // #ifndef CARPETX_CARPETX_IO_SILO_HXX
