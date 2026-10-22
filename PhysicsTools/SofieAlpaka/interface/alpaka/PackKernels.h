#ifndef PhysicsTools_SofieAlpaka_interface_alpaka_PackKernels_h
#define PhysicsTools_SofieAlpaka_interface_alpaka_PackKernels_h

#include <cstddef>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/SofieAlpaka/interface/PackDescriptor.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sofie::detail {

  // Gather the elements of a strided SoA tensor into a contiguous, row-major buffer.
  void pack(Queue& queue, cms::sofie::alpakatools::PackDescriptor const& desc, std::byte const* soa, std::byte* buffer);

  // Scatter the elements of a contiguous, row-major buffer into a strided SoA tensor.
  void unpack(Queue& queue, cms::sofie::alpakatools::PackDescriptor const& desc, std::byte const* buffer, std::byte* soa);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sofie::detail

#endif  // PhysicsTools_SofieAlpaka_interface_alpaka_PackKernels_h
