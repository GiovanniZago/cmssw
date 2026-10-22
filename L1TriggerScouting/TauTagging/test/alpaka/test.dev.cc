#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "L1TriggerScouting/TauTagging/interface/alpaka/SofieSTTRelu.hxx"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Smoke test for the SOFIE STT model as used by SoftTauIdML:
  //   input 0: features tensor, [2048 clusters][16 constituents][10 features] = 327680 floats
  //   input 1: padding mask,   [2048 clusters][16 constituents]             =  32768 floats
  //   outputs: class_logit, charge_logit, reg_pt, one float per cluster (2048)
  class TestTauTagging : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestTauTagging);
    CPPUNIT_TEST(testDeviceBuffers);
    CPPUNIT_TEST_SUITE_END();

  public:
    void testDeviceBuffers();
  };

  CPPUNIT_TEST_SUITE_REGISTRATION(TestTauTagging);

  void TestTauTagging::testDeviceBuffers() {
    // create alpaka stuff
    const Device device = alpaka::getDevByIdx<Platform>(Platform{}, 0);
    Queue queue(device);
    Event EvInferenceDone(alpaka::getDev(queue));
    Event EvFeaturesReady(alpaka::getDev(queue));

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    // setup the model
    std::string weightFile = "/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_fp32_b2048_s16_GPU_ALPAKA.dat";
    // the SOFIE Sessions are templated on an alpaka *tag*: derive the one matching
    // the current backend accelerator
    SOFIE_STT_ort_relu_fp32_b2048_s16::Session<alpaka::AccToTag<Acc1D>> session(weightFile);
    alpaka::wait(session.queue);
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> fdist(-100.0f, 100.0f);

    // The cms::alpakatools buffer helpers use a 32-bit index type, while the SOFIE
    // BufF1D is indexed by std::size_t: allocate the device buffers and the host
    // copy views with the SOFIE index type so infer() accepts them.
    using SofieIdx = std::size_t;
    const alpaka::DevCpu hostDev = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);

    // Input 0: float[327680], uniform in [-1, 1]
    auto input_h_0 = cms::alpakatools::make_host_buffer<float[]>(queue, 327680);
    {
      auto* p = reinterpret_cast<float*>(alpaka::getPtrNative(input_h_0));
      for (size_t k = 0; k < 327680; ++k)
        p[k] = static_cast<float>(fdist(rng));
    }
    // in the following why not cms::alpakatools::make_device_buffer
    auto input_d_0 = alpaka::allocBuf<float, SofieIdx>(alpaka::getDev(queue), static_cast<SofieIdx>(327680));
    auto input_hv_0 = alpaka::createView(hostDev, alpaka::getPtrNative(input_h_0), static_cast<SofieIdx>(327680));
    alpaka::memcpy(queue, input_d_0, input_hv_0);

    // Input 1: float[32768], all ones
    auto input_h_1 = cms::alpakatools::make_host_buffer<float[]>(queue, 32768);
    {
      auto* p = reinterpret_cast<float*>(alpaka::getPtrNative(input_h_1));
      std::fill(p, p + 32768, static_cast<float>(1));
    }
    auto input_d_1 = alpaka::allocBuf<float, SofieIdx>(alpaka::getDev(queue), static_cast<SofieIdx>(32768));
    auto input_hv_1 = alpaka::createView(hostDev, alpaka::getPtrNative(input_h_1), static_cast<SofieIdx>(32768));
    alpaka::memcpy(queue, input_d_1, input_hv_1);

    // The generated SOFIE session currently binds against the GPU sofieBLAS backends
    // only: with TagCpuSerial SofieSTT.hxx does not compile (its BlasCpu wants a
    // QueueCpuBlocking&, and BlasCpu::gemmStridedBatched has a different signature).
    // Until the model is regenerated, run the inference only on the GPU backends;
    // on serial this test still exercises the buffer plumbing above.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    // The SOFIE session runs the inference on its own internal queue, which is NOT
    // ordered with `queue`: the host-to-device input copies must be complete
    // before the session reads the input buffers.
    alpaka::enqueue(queue, EvFeaturesReady);
    alpaka::wait(session.queue, EvFeaturesReady);

    // run the model
    auto out_d = session.infer_nosync(input_d_0, input_d_1);
    out_d = session.infer_nosync(input_d_0, input_d_1);
    out_d = session.infer_nosync(input_d_0, input_d_1);
    alpaka::enqueue(session.queue, EvInferenceDone);
    alpaka::wait(queue, EvInferenceDone);

    // copy the results to host (note the SOFIE infer() order: class_logit, charge_logit, reg_pt)
    // Output 0: float[2048]
    auto out_h_0 = cms::alpakatools::make_host_buffer<float[]>(queue, 2048);
    // Output 1: float[2048]
    auto out_h_1 = cms::alpakatools::make_host_buffer<float[]>(queue, 2048);
    // Output 2: float[2048]
    auto out_h_2 = cms::alpakatools::make_host_buffer<float[]>(queue, 2048);

    // copy the results to host
    auto out_hv_0 =
        alpaka::createView(hostDev, alpaka::getPtrNative(out_h_0), static_cast<SofieIdx>(2048));
    auto out_hv_1 =
        alpaka::createView(hostDev, alpaka::getPtrNative(out_h_1), static_cast<SofieIdx>(2048));
    auto out_hv_2 =
        alpaka::createView(hostDev, alpaka::getPtrNative(out_h_2), static_cast<SofieIdx>(2048));
    alpaka::memcpy(queue, out_hv_0, std::get<0>(out_d));
    alpaka::memcpy(queue, out_hv_1, std::get<1>(out_d));
    alpaka::memcpy(queue, out_hv_2, std::get<2>(out_d));
    alpaka::wait(queue);

    // check the outputs are well-formed
    const float* cls = reinterpret_cast<const float*>(alpaka::getPtrNative(out_h_0));
    const float* charge = reinterpret_cast<const float*>(alpaka::getPtrNative(out_h_1));
    const float* pt = reinterpret_cast<const float*>(alpaka::getPtrNative(out_h_2));
    for (size_t k = 0; k < 2048; ++k) {
      CPPUNIT_ASSERT(std::isfinite(cls[k]));
      CPPUNIT_ASSERT(std::isfinite(charge[k]));
      CPPUNIT_ASSERT(std::isfinite(pt[k]));
    }

    // print the inference outputs for an easy visual inspection
    std::cout << "\n[session] SOFIE unified memory pool: " << std::fixed << std::setprecision(2)
              << static_cast<double>(session.GetIntermediateMemoryPoolSize()) / (1024.0 * 1024.0) << " MB\n";

    constexpr size_t nShow = 8;
    std::cout << std::string(70, '-') << '\n'
              << std::setw(7) << "cluster" << std::setw(15) << "cls (logit)" << std::setw(15) << "charge (logit)"
              << std::setw(20) << "pt (regression)" << '\n'
              << std::string(70, '-') << '\n';
    for (size_t k = 0; k < nShow; ++k)
      std::cout << std::setw(7) << k << std::setw(15) << std::setprecision(5) << cls[k] << std::setw(15) << charge[k]
                << std::setw(20) << std::setprecision(5) << pt[k] << '\n';
    std::cout << std::string(70, '-') << std::endl;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
