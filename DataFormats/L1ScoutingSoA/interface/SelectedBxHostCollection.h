#ifndef DataFormats_L1ScoutingSoA_interface_SelectedBxHostCollection_h
#define DataFormats_L1ScoutingSoA_interface_SelectedBxHostCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/L1ScoutingSoA/interface/SelectedBxSoA.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace l1sc {

using SelectedBxHostCollection = PortableHostCollection<SelectedBxSoA>;

}  // namespace l1sc

#endif  // DataFormats_L1ScoutingSoA_interface_SelectedBxHostCollection_h