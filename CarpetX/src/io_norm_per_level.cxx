// Per-level norms and fp16 admissibility output (out_norm_per_level).
//
// For every group named in out_norm_vars and every refinement level, one
// row per output iteration in <out_dir>/norms/level<N>/<group>.tsv:
//   iteration time <var>.l2 <var>.linf <var>.min <var>.max <var>.npoints
//   <var>.nsubnormal16 <var>.admiss <var>.admiss_shell ...
// where l2 is a plain point average sqrt(sum u^2 / npoints) over the
// level's interior (no ghost zones; unlike io_norm.cxx's volume-weighted
// hierarchy reduction there is no half weight at box faces or refinement
// boundaries, so a vertex-centered point shared by two boxes is counted
// once per box), linf the max |u|, nsubnormal16 the number of points with
// 0 < |u| below binary16's minimum normal 2^-14, and admiss the
// fine-masked maximum of q(|u|) / max(|D2u|, floor) defined in
// reduction.cxx (admiss_shell: the same over the shell around
// out_norm_admiss_radius). Points where out_norm_mask_var is below
// out_norm_mask_below are excluded from every column. The io.cxx hook
// calls this on every iteration out_norm_per_level_every divides and on
// the iteration before each, so a regrid is bracketed by two rows.

#include "io_norm.hxx"

#include "driver.hxx"
#include "io_meta.hxx"
#include "reduction.hxx"
#include "timer.hxx"

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

namespace {

std::string level_dir(const char *const out_dir, const int level) {
  std::ostringstream buf;
  buf << out_dir << "/norms/level" << level;
  return buf.str();
}

std::string group_filename(const char *const out_dir, const int level,
                           const int gi) {
  std::string groupname = CCTK_FullGroupName(gi);
  groupname = std::regex_replace(groupname, std::regex("::"), "-");
  for (auto &ch : groupname)
    ch = std::tolower(ch);
  return level_dir(out_dir, level) + "/" + groupname + ".tsv";
}

// -inf (no contributing point) is written as NaN
CCTK_REAL nan_if_unset(const CCTK_REAL x) {
  return std::isinf(x) && x < 0 ? std::numeric_limits<CCTK_REAL>::quiet_NaN()
                                : x;
}

} // namespace

void OutputNormsPerLevel(const cGH *restrict cctkGH) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (!out_norm_per_level)
    return;
  if (out_norm_vars[0] == '\0')
    return;

  static Timer timer("OutputNormsPerLevel");
  Interval interval(timer);

  // Find output groups (same selection as OutputNorms)
  const std::vector<bool> group_enabled = [&] {
    std::vector<bool> enabled(CCTK_NumGroups(), false);
    const auto callback{
        [](const int index, const char *const optstring, void *const arg) {
          std::vector<bool> &enabled = *static_cast<std::vector<bool> *>(arg);
          enabled.at(CCTK_GroupIndexFromVarI(index)) = true;
        }};
    CCTK_TraverseString(out_norm_vars, callback, &enabled, CCTK_GROUP_OR_VAR);
    return enabled;
  }();
  const auto num_out_groups =
      std::count(group_enabled.begin(), group_enabled.end(), true);
  if (num_out_groups == 0)
    return;

  static std::vector<bool> previous_group_enabled;
  const bool group_enabled_changed = group_enabled != previous_group_enabled;
  if (group_enabled_changed)
    previous_group_enabled = group_enabled;

  // Mask variable, resolved once
  static int mask_vi = -2;
  if (mask_vi == -2) {
    if (out_norm_mask_var[0] == '\0') {
      mask_vi = -1;
    } else {
      mask_vi = CCTK_VarIndex(out_norm_mask_var);
      if (mask_vi < 0)
        CCTK_VERROR("out_norm_mask_var \"%s\" is not a known variable",
                    out_norm_mask_var);
      if (CCTK_GroupTypeFromVarI(mask_vi) != CCTK_GF)
        CCTK_VERROR("out_norm_mask_var \"%s\" is not a grid function",
                    out_norm_mask_var);
    }
  }

  const bool is_root = CCTK_MyProc(nullptr) == 0;
  const std::string sep = "\t";
  const int tl = 0;
  const int numgroups = CCTK_NumGroups();
  const int nlevels = ghext->num_levels();
  const std::vector<std::string> columns = {
      "l2",           "linf",   "min",         "max", "npoints",
      "nsubnormal16", "admiss", "admiss_shell"};

  // Directories, created once per level (levels appear as the run regrids)
  static std::set<int> created_levels;
  if (is_root) {
    static std::once_flag create_directory;
    std::call_once(create_directory, [&]() {
      const int mode = 0755;
      int ierr = CCTK_CreateDirectory(mode, out_dir);
      assert(ierr >= 0);
      std::ostringstream buf;
      buf << out_dir << "/norms";
      ierr = CCTK_CreateDirectory(mode, buf.str().c_str());
      assert(ierr >= 0);
    });
    for (int level = 0; level < nlevels; ++level) {
      if (created_levels.count(level))
        continue;
      const int ierr =
          CCTK_CreateDirectory(0755, level_dir(out_dir, level).c_str());
      assert(ierr >= 0);
      created_levels.insert(level);
    }
  }

  for (int gi = 0; gi < numgroups; ++gi) {
    if (!group_enabled.at(gi))
      continue;
    if (CCTK_GroupTypeI(gi) != CCTK_GF)
      continue;

    for (int level = 0; level < nlevels; ++level) {
      // Level 0 of patch 0 decides validity, as in OutputNorms; a level a
      // patch lacks is skipped inside reduce_level
      const int patch = 0;
      const GHExt::PatchData &restrict patchdata = ghext->patchdata.at(patch);
      if (level >= int(patchdata.leveldata.size()))
        continue;
      const GHExt::PatchData::LevelData &restrict leveldata =
          patchdata.leveldata.at(level);
      const GHExt::PatchData::LevelData::GroupData &restrict groupdata =
          *leveldata.groupdata.at(gi);

      std::vector<int> valid_vars;
      for (int vi = 0; vi < groupdata.numvars; ++vi)
        if (groupdata.valid.at(tl).at(vi).get().valid_int)
          valid_vars.push_back(vi);
      if (valid_vars.empty())
        continue;

      const std::string filename = group_filename(out_dir, level, gi);

      std::ofstream file;
      output_file_description_t ofd;
      if (is_root) {
        // Header when the group set changed or the file is new (a level
        // created by a later regrid gets its own header)
        const bool file_exists = std::ifstream(filename).good();
        file.open(filename, std::ios_base::app);
        ofd.filename = filename;
        if (group_enabled_changed || !file_exists) {
          int col = 0;
          file << "# " << ++col << ":iteration";
          file << sep << ++col << ":time";
          for (const int vi : valid_vars) {
            std::string varname =
                CCTK_FullVarName(groupdata.firstvarindex + vi);
            for (auto &ch : varname)
              ch = std::tolower(ch);
            for (const auto &column : columns)
              file << sep << ++col << ":" << varname << "." << column;
          }
          file << "\n";
        }
        file << std::setprecision(std::numeric_limits<CCTK_REAL>::digits10 + 1)
             << std::scientific;
        file << cctk_iteration << sep << cctk_time;
      }

      for (const int vi : valid_vars) {
        ofd.variables.push_back(CCTK_FullVarName(groupdata.firstvarindex + vi));

        const level_reduction_t red = reduce_level(gi, vi, tl, level, mask_vi);

        if (is_root) {
          const CCTK_REAL nan = std::numeric_limits<CCTK_REAL>::quiet_NaN();
          file << sep << red.norm2() << sep << red.maxabs << sep << red.min
               << sep << red.max << sep << red.npoints << sep
               << red.nsubnormal16 << sep
               << (red.has_admiss ? nan_if_unset(red.admiss) : nan) << sep
               << (red.has_admiss ? nan_if_unset(red.admiss_shell) : nan);
        }
      }

      if (is_root) {
        file << "\n";
        file.close();

        ofd.description = "CarpetX per-level TSV norms output";
        ofd.writer_thorn = CCTK_THORNSTRING;
        ofd.iterations = {cctk_iteration};
        ofd.reductions = {
            reduction_t::norm2,
            reduction_t::norm_inf,
            reduction_t::minimum,
            reduction_t::maximum,
        };
        ofd.format_name = "CarpetX/norms-per-level/TSV";
        ofd.format_version = {1, 0, 0};

        OutputMeta_RegisterOutputFile(std::move(ofd));
      }
    }
  }
}

} // namespace CarpetX
