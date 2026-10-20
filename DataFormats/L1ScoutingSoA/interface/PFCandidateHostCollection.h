#ifndef DataFormats_L1ScoutingSoA_interface_PFCandidateHostCollection_h
#define DataFormats_L1ScoutingSoA_interface_PFCandidateHostCollection_h

#include "DataFormats/L1ScoutingSoA/interface/PFCandidateSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace l1sc {

using PFCandidateHostCollection = PortableHostCollection<PFCandidateSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_PFCandidateHostCollection_h