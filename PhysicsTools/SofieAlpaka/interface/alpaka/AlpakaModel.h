#ifndef PhysicsTools_SofieAlpaka_interface_alpaka_Model_h
#define PhysicsTools_SofieAlpaka_interface_alpaka_Model_h

// SOFIE session running the inference directly on SoA data in alpaka device memory.
// Adapted from PhysicsTools/ONNXRuntimeAlpaka/interface/alpaka/AlpakaSession.h .
//
// The wrapper drives a SOFIE-generated Session loaded from its weights file. The pack and
// inference work runs on the session's own queue (TSession::queue); if that queue differs
// from the queue given to forward(), the two are bridged with events so that
//   - the SoA data given as input is complete before any pack/reading kernel touches it, and
//   - the event queue resumes only once the inference has written the output views.
// All copies happen without blocking the host, except for the wait performed inside the
// generated TSession::infer(std::span, std::span) entry point itself.

#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "PhysicsTools/SofieAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/SofieAlpaka/interface/alpaka/PackKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sofie {

  // TSession is a SOFIE-generated Session struct, e.g.
  //   SOFIE_STT_ort_relu_fp32_b2048_s16::Session<alpaka::AccToTag<Acc1D>>
  // It is expected to expose:
  //   - a constructor Session(std::string weights_file);
  //   - a public member 'queue';
  //   - view aliases ViewConstF1D and ViewF1D (non-owning float* views into the tensors);
  //   - infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs).
  //
  // A Model is not thread safe: instantiate one per EDM stream. Construct it in beginStream():
  // the Session constructor reads the weights file and allocates the device buffers.
  template <typename TSession>
  class Model {
  public:
    explicit Model(std::string weights_path) : weights_path_(std::move(weights_path)), session_(weights_path_) {}

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Forward pass (inference) of the model on registered SoA tensors.
    // Inputs and outputs are matched positionally; the outputs are written directly into the SoA
    // memory when contiguous (unpacked from a staging buffer otherwise), exactly like in
    // PhysicsTools/ONNXRuntimeAlpaka.
    void forward(Queue& queue,
                 cms::sofie::alpakatools::TensorCollection<Queue>& inputs,
                 cms::sofie::alpakatools::TensorCollection<Queue>& outputs) {
      if (alpaka::getDev(queue) != alpaka::getDev(session_.queue)) {
        throw cms::Exception("LogicError") << "SOFIE model " << weights_path_
                                           << " used on a different device than the one of its session queue.";
      }

      // empty tensors carry no work
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].batch_size() == 0)
          return;
      }

      // everything SOFIE does runs on the session's own queue
      auto& run_queue = session_.queue;
      const bool bridge = (queue != run_queue);
      if (bridge) {
        if (not ready_)
          ready_.emplace(alpaka::getDev(queue));
        alpaka::enqueue(queue, *ready_);
        alpaka::wait(run_queue, *ready_);
      }

      // wrap the SoA memory in the view types the Session consumes, packing it into staging
      // buffers where the strided SoA layout prevents a direct view
      std::vector<Staging> staging(inputs.size() + outputs.size());

      std::vector<typename TSession::ViewConstF1D> input_views;
      input_views.reserve(inputs.size());
      for (size_t i = 0; i < inputs.size(); ++i)
        input_views.emplace_back(stage_in(run_queue, inputs[i], staging[i]));

      std::vector<typename TSession::ViewF1D> output_views;
      output_views.reserve(outputs.size());
      for (size_t i = 0; i < outputs.size(); ++i)
        output_views.emplace_back(stage_out(run_queue, outputs[i], staging[inputs.size() + i]));

      // run the inference: writes the model outputs into the views of the SoA memory (or of the
      // staging buffers) prepared above
      session_.infer(std::span<typename TSession::ViewConstF1D const>(input_views),
                     std::span<typename TSession::ViewF1D>(output_views));

      // scatter the staged outputs back into the strided SoA memory, where needed
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto& st = staging[inputs.size() + i];
        if (st.device)
          detail::unpack(run_queue,
                         outputs[i].pack_descriptor(),
                         st.device->data(),
                         static_cast<std::byte*>(outputs[i].data()));
      }

      // let the event queue resume only when the inference is complete
      if (bridge) {
        if (not done_)
          done_.emplace(alpaka::getDev(run_queue));
        alpaka::enqueue(run_queue, *done_);
        alpaka::wait(queue, *done_);
      }
    }

  private:
    // temporary buffer used to present a strided tensor to / from the SOFIE session
    struct Staging {
      std::optional<cms::alpakatools::device_buffer<Device, std::byte[]>> device;
    };

    static size_t elements(cms::sofie::alpakatools::detail::ITensorHandle const& tensor) {
      return tensor.shape_bytes() / sizeof(float);
    }

    // throw unless the registered block holds floats; SOFIE is statically typed on float and
    // silently reinterpreting other element types would corrupt the inference
    void check_float(const cms::sofie::alpakatools::detail::ITensorHandle& tensor) const {
      if (tensor.bytes() != sizeof(float)) {
        throw cms::Exception("InvalidArgument")
            << "SOFIE model " << weights_path_ << " expects float tensors, but a tensor of "
            << tensor.bytes() << "-byte elements was registered.";
      }
    }

    template <typename TView, typename TElem>
    static TView make_view(Queue& queue, std::byte* data, size_t n_elements) {
      using Idx = typename TSession::Idx;
      return TView{reinterpret_cast<TElem*>(data), alpaka::getDev(queue),
                   typename TSession::Ext1D{Idx{n_elements}}};
    }

    // return a const view of the registered input; pack it into a staging buffer first if the
    // strided SoA layout prevents a direct view
    typename TSession::ViewConstF1D stage_in(Queue& queue,
                                             cms::sofie::alpakatools::detail::ITensorHandle const& tensor,
                                             Staging& staging) const {
      // check_float(tensor);
      std::byte* data = static_cast<std::byte*>(tensor.data());
      if (not tensor.is_contiguous()) {
        staging.device = cms::alpakatools::make_device_buffer<std::byte[]>(queue, tensor.shape_bytes());
        detail::pack(queue, tensor.pack_descriptor(), data, staging.device->data());
        data = staging.device->data();
      }
      return make_view<typename TSession::ViewConstF1D, const float>(queue, data, elements(tensor));
    }

    // return a writable view of the registered output, placed in a staging buffer (and later
    // scattered back by unpack) when the strided SoA layout prevents a direct view
    typename TSession::ViewF1D stage_out(Queue& queue,
                                         cms::sofie::alpakatools::detail::ITensorHandle const& tensor,
                                         Staging& staging) const {
      check_float(tensor);
      std::byte* data = static_cast<std::byte*>(tensor.data());
      if (not tensor.is_contiguous()) {
        staging.device = cms::alpakatools::make_device_buffer<std::byte[]>(queue, tensor.shape_bytes());
        data = staging.device->data();
      }
      return make_view<typename TSession::ViewF1D, float>(queue, data, elements(tensor));
    }

    std::string weights_path_;
    TSession session_;

    std::optional<Event> ready_;
    std::optional<Event> done_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sofie

#endif  // PhysicsTools_SofieAlpaka_interface_alpaka_Model_h
