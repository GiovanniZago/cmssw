#ifndef L1TriggerScouting_TauTagging_interface_SoftTauIdMLCommon_h
#define L1TriggerScouting_TauTagging_interface_SoftTauIdMLCommon_h

#include <compare>
#include <cstdint>
#include <string_view>
#include "FWCore/Utilities/interface/Exception.h"

namespace l1sc {

enum class Substep : uint8_t { Sorting = 0, Reshaping = 1, Tagging = 2 };

inline Substep parseSubstep(std::string_view value) {
  if (value == "sorting") {
    return Substep::Sorting;
  }
  if (value == "reshaping") {
    return Substep::Reshaping;
  }
  if (value == "tagging") {
    return Substep::Tagging;
  }

  throw cms::Exception("Configuration")
      << "Invalid substep value '" << value << "'. "
      << "Allowed values are: sorting, reshaping, tagging.";
}

}  // namespace l1sc

#endif  // L1TriggerScouting_TauTagging_interface_SoftTauIdMLCommon_h