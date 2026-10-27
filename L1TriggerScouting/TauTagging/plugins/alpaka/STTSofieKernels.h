#ifndef L1TriggerScouting_TauTagging_plugins_alpaka_STTSofieKernels_h
#define L1TriggerScouting_TauTagging_plugins_alpaka_STTSofieKernels_h

#include <algorithm>
#include <type_traits>
#include <memory>
#include <cmath>
#include <vector>
#include <alpaka/alpaka.hpp>
#include <sofieBLAS/sofieBLAS.hpp>
#include <span>
#include "SOFIE/SOFIE_common.hxx"
#include <fstream>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#ifndef SOFIE_ALPAKA_DIM1D_DEFINED
#define SOFIE_ALPAKA_DIM1D_DEFINED
using Dim1D = alpaka::DimInt<1>;
#endif // SOFIE_ALPAKA_DIM1D_DEFINED

#ifndef SOFIE_ALPAKA_WORKDIV_DEFINED
#define SOFIE_ALPAKA_WORKDIV_DEFINED
template<typename TDim, typename TIdx>
inline alpaka::WorkDivMembers<TDim, TIdx> sofie_workdiv(
    alpaka::Vec<TDim, TIdx> const& numElems, TIdx blockSz = TIdx{256})
{
    auto const numBlocks = alpaka::Vec<TDim, TIdx>::all(
        std::max(TIdx{1}, (numElems[0] + blockSz - TIdx{1}) / blockSz));
    return alpaka::WorkDivMembers<TDim, TIdx>(
        numBlocks,
        alpaka::Vec<TDim, TIdx>::all(blockSz),
        alpaka::Vec<TDim, TIdx>::all(TIdx{1}));
}
#endif // SOFIE_ALPAKA_WORKDIV_DEFINED

#ifndef SOFIE_ALPAKA_DEVICE_MATH_DEFINED
#define SOFIE_ALPAKA_DEVICE_MATH_DEFINED
#if defined(__CUDA_ARCH__)
#define SOFIE_DEVICE_exp(acc, ...) ::exp(__VA_ARGS__)
#define SOFIE_DEVICE_sqrt(acc, ...) ::sqrt(__VA_ARGS__)
#define SOFIE_DEVICE_log(acc, ...) ::log(__VA_ARGS__)
#define SOFIE_DEVICE_sin(acc, ...) ::sin(__VA_ARGS__)
#define SOFIE_DEVICE_cos(acc, ...) ::cos(__VA_ARGS__)
#define SOFIE_DEVICE_abs(acc, ...) ::abs(__VA_ARGS__)
#define SOFIE_DEVICE_pow(acc, ...) ::pow(__VA_ARGS__)
#define SOFIE_DEVICE_atan2(acc, ...) ::atan2(__VA_ARGS__)
#define SOFIE_DEVICE_atan(acc, ...) ::atan(__VA_ARGS__)
#define SOFIE_DEVICE_floor(acc, ...) ::floor(__VA_ARGS__)
#define SOFIE_DEVICE_ceil(acc, ...) ::ceil(__VA_ARGS__)
#define SOFIE_DEVICE_round(acc, ...) ::round(__VA_ARGS__)
#define SOFIE_DEVICE_cbrt(acc, ...) ::cbrt(__VA_ARGS__)
#define SOFIE_DEVICE_tanh(acc, ...) ::tanh(__VA_ARGS__)
#define SOFIE_DEVICE_sinh(acc, ...) ::sinh(__VA_ARGS__)
#define SOFIE_DEVICE_cosh(acc, ...) ::cosh(__VA_ARGS__)
#else
#define SOFIE_DEVICE_exp(acc, ...) exp(acc, __VA_ARGS__)
#define SOFIE_DEVICE_sqrt(acc, ...) sqrt(acc, __VA_ARGS__)
#define SOFIE_DEVICE_log(acc, ...) log(acc, __VA_ARGS__)
#define SOFIE_DEVICE_sin(acc, ...) sin(acc, __VA_ARGS__)
#define SOFIE_DEVICE_cos(acc, ...) cos(acc, __VA_ARGS__)
#define SOFIE_DEVICE_abs(acc, ...) abs(acc, __VA_ARGS__)
#define SOFIE_DEVICE_pow(acc, ...) pow(acc, __VA_ARGS__)
#define SOFIE_DEVICE_atan2(acc, ...) atan2(acc, __VA_ARGS__)
#define SOFIE_DEVICE_atan(acc, ...) atan(acc, __VA_ARGS__)
#define SOFIE_DEVICE_floor(acc, ...) floor(acc, __VA_ARGS__)
#define SOFIE_DEVICE_ceil(acc, ...) ceil(acc, __VA_ARGS__)
#define SOFIE_DEVICE_round(acc, ...) round(acc, __VA_ARGS__)
#define SOFIE_DEVICE_cbrt(acc, ...) cbrt(acc, __VA_ARGS__)
#define SOFIE_DEVICE_tanh(acc, ...) tanh(acc, __VA_ARGS__)
#define SOFIE_DEVICE_sinh(acc, ...) sinh(acc, __VA_ARGS__)
#define SOFIE_DEVICE_cosh(acc, ...) cosh(acc, __VA_ARGS__)
#endif // __CUDA_ARCH__
#endif // SOFIE_ALPAKA_DEVICE_MATH_DEFINED

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::SOFIE_STT_ort_relu_fp32_b2048_s16 {

//--- ALPAKA Kernels

//------ LAYERNORM_KERNEL_ALPAKA
   struct LayerNormKernel_op_3 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ X,
         T const* __restrict__ scale,
         T const* __restrict__ bias,
         T* __restrict__ Y,
         std::size_t const axesLength) const {

         // Block-parallel LayerNorm: one block per row, 16 threads per block, 1 item(s) per thread, 10 active elements.
         auto& shmem = alpaka::declareSharedVar<T[16], __COUNTER__>(acc);
         auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
         auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
         if (row >= axesLength) return;

         std::size_t const axis_0 = (row / 16u) % 2048u;
         std::size_t const axis_1 = (row / 1u) % 16u;

         std::size_t const row_base =
            axis_0 * 160u +
            axis_1 * 10u;

         constexpr std::size_t itemsPerThread = 1u;
         T vals[itemsPerThread];
         std::size_t idxs[itemsPerThread];
         T local_sum = static_cast<T>(0);
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 16u;
            if (j < 10u) {
               std::size_t const norm_offset = j * 1u;
               std::size_t const s_norm_offset = j * 1u;
               std::size_t const b_norm_offset = j * 1u;
               idxs[k] = row_base + norm_offset;
               vals[k] = X[idxs[k]];
               local_sum += vals[k];
            } else {
               idxs[k] = 0u;
               vals[k] = static_cast<T>(0);
            }
         }

         // Pass 1: compute mean via shared-memory tree reduction
         shmem[tid] = local_sum;
         alpaka::syncBlockThreads(acc);
         if (tid < 8u) shmem[tid] += shmem[tid + 8u];
         alpaka::syncBlockThreads(acc);
         if (tid < 4u) shmem[tid] += shmem[tid + 4u];
         alpaka::syncBlockThreads(acc);
         if (tid < 2u) shmem[tid] += shmem[tid + 2u];
         alpaka::syncBlockThreads(acc);
         if (tid < 1u) shmem[tid] += shmem[tid + 1u];
         alpaka::syncBlockThreads(acc);
         T const mean = shmem[0] / static_cast<T>(10);
         alpaka::syncBlockThreads(acc);

         // Pass 2: compute variance
         T local_sq = static_cast<T>(0);
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 16u;
            if (j < 10u) { T const diff = vals[k] - mean; local_sq += diff * diff; }
         }
         shmem[tid] = local_sq;
         alpaka::syncBlockThreads(acc);
         if (tid < 8u) shmem[tid] += shmem[tid + 8u];
         alpaka::syncBlockThreads(acc);
         if (tid < 4u) shmem[tid] += shmem[tid + 4u];
         alpaka::syncBlockThreads(acc);
         if (tid < 2u) shmem[tid] += shmem[tid + 2u];
         alpaka::syncBlockThreads(acc);
         if (tid < 1u) shmem[tid] += shmem[tid + 1u];
         alpaka::syncBlockThreads(acc);
         T const invStdDev = static_cast<T>(1) / sqrt(acc, shmem[0] / static_cast<T>(10) + static_cast<T>(0.000010));


         // Pass 3: normalize + scale + bias
         std::size_t const scale_base =
            0u;
         std::size_t const bias_base =
            0u;
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 16u;
            if (j < 10u) {
               std::size_t const norm_offset = j * 1u;
               std::size_t const s_norm_offset = j * 1u;
               std::size_t const b_norm_offset = j * 1u;
               T out_val = scale[scale_base + s_norm_offset] * invStdDev * (vals[k] - mean);
               out_val += bias[bias_base + b_norm_offset];
               Y[idxs[k]] = out_val;
            }
         }
      }
   };

//------ GEMM_BIAS_ADD_KERNEL_ALPAKA
struct GemmBiasAddKernel {
   template<typename TAcc, typename T>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, T* __restrict__ Y, T const* __restrict__ C, std::size_t m, std::size_t n, std::size_t cRows, std::size_t cCols, bool hasExtraC, bool applyRelu, std::size_t totalElements) const {
      auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx >= totalElements) return;
      std::size_t sliceSize = m * n;
      std::size_t i = idx / sliceSize;
      std::size_t within = idx % sliceSize;
      std::size_t r = within / n;
      std::size_t c = within % n;
      std::size_t cSliceOffset = hasExtraC ? i * (cRows * cCols) : 0;
      std::size_t cr = (cRows == 1) ? 0 : r;
      std::size_t cc = (cCols == 1) ? 0 : c;
      T value = Y[idx] + C[cSliceOffset + cr * cCols + cc];
      Y[idx] = applyRelu ? (value >= T(0) ? value : T(0)) : value;
   }
};

//------ RELU_KERNEL_ALPAKA
struct ReluKernel {
   template<typename TAcc, typename T>
   ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {
         auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
         if (idx < numElements) {
         out[idx] = data[idx] >= T(0) ? data[idx] : 0;
      }
   }
};

//------ LAYERNORM_KERNEL_ALPAKA
   struct LayerNormKernel_op_6 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ X,
         T const* __restrict__ scale,
         T const* __restrict__ bias,
         T* __restrict__ Y,
         std::size_t const axesLength) const {

         // Block-parallel LayerNorm: one block per row, 128 threads per block, 1 item(s) per thread, 128 active elements.
         auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);
         auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
         auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
         if (row >= axesLength) return;

         std::size_t const axis_0 = (row / 16u) % 2048u;
         std::size_t const axis_1 = (row / 1u) % 16u;

         std::size_t const row_base =
            axis_0 * 2048u +
            axis_1 * 128u;

         constexpr std::size_t itemsPerThread = 1u;
         T vals[itemsPerThread];
         std::size_t idxs[itemsPerThread];
         T local_sum = static_cast<T>(0);
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 128u;
            if (j < 128u) {
               std::size_t const norm_offset = j * 1u;
               std::size_t const s_norm_offset = j * 1u;
               std::size_t const b_norm_offset = j * 1u;
               idxs[k] = row_base + norm_offset;
               vals[k] = X[idxs[k]];
               local_sum += vals[k];
            } else {
               idxs[k] = 0u;
               vals[k] = static_cast<T>(0);
            }
         }

         // Pass 1: compute mean via shared-memory tree reduction
         shmem[tid] = local_sum;
         alpaka::syncBlockThreads(acc);
         if (tid < 64u) shmem[tid] += shmem[tid + 64u];
         alpaka::syncBlockThreads(acc);
         if (tid < 32u) shmem[tid] += shmem[tid + 32u];
         alpaka::syncBlockThreads(acc);
         if (tid < 16u) shmem[tid] += shmem[tid + 16u];
         alpaka::syncBlockThreads(acc);
         if (tid < 8u) shmem[tid] += shmem[tid + 8u];
         alpaka::syncBlockThreads(acc);
         if (tid < 4u) shmem[tid] += shmem[tid + 4u];
         alpaka::syncBlockThreads(acc);
         if (tid < 2u) shmem[tid] += shmem[tid + 2u];
         alpaka::syncBlockThreads(acc);
         if (tid < 1u) shmem[tid] += shmem[tid + 1u];
         alpaka::syncBlockThreads(acc);
         T const mean = shmem[0] / static_cast<T>(128);
         alpaka::syncBlockThreads(acc);

         // Pass 2: compute variance
         T local_sq = static_cast<T>(0);
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 128u;
            if (j < 128u) { T const diff = vals[k] - mean; local_sq += diff * diff; }
         }
         shmem[tid] = local_sq;
         alpaka::syncBlockThreads(acc);
         if (tid < 64u) shmem[tid] += shmem[tid + 64u];
         alpaka::syncBlockThreads(acc);
         if (tid < 32u) shmem[tid] += shmem[tid + 32u];
         alpaka::syncBlockThreads(acc);
         if (tid < 16u) shmem[tid] += shmem[tid + 16u];
         alpaka::syncBlockThreads(acc);
         if (tid < 8u) shmem[tid] += shmem[tid + 8u];
         alpaka::syncBlockThreads(acc);
         if (tid < 4u) shmem[tid] += shmem[tid + 4u];
         alpaka::syncBlockThreads(acc);
         if (tid < 2u) shmem[tid] += shmem[tid + 2u];
         alpaka::syncBlockThreads(acc);
         if (tid < 1u) shmem[tid] += shmem[tid + 1u];
         alpaka::syncBlockThreads(acc);
         T const invStdDev = static_cast<T>(1) / sqrt(acc, shmem[0] / static_cast<T>(128) + static_cast<T>(0.000010));


         // Pass 3: normalize + scale + bias
         std::size_t const scale_base =
            0u;
         std::size_t const bias_base =
            0u;
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 128u;
            if (j < 128u) {
               std::size_t const norm_offset = j * 1u;
               std::size_t const s_norm_offset = j * 1u;
               std::size_t const b_norm_offset = j * 1u;
               T out_val = scale[scale_base + s_norm_offset] * invStdDev * (vals[k] - mean);
               out_val += bias[bias_base + b_norm_offset];
               Y[idxs[k]] = out_val;
            }
         }
      }
   };

//------ LAYERNORM_KERNEL_ALPAKA
   struct LayerNormKernel_op_9 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ X,
         T const* __restrict__ scale,
         T const* __restrict__ bias,
         T* __restrict__ Y,
         std::size_t const axesLength) const {

         // Block-parallel LayerNorm: one block per row, 256 threads per block, 2 item(s) per thread, 512 active elements.
         auto& shmem = alpaka::declareSharedVar<T[256], __COUNTER__>(acc);
         auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
         auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
         if (row >= axesLength) return;

         std::size_t const axis_0 = (row / 16u) % 2048u;
         std::size_t const axis_1 = (row / 1u) % 16u;

         std::size_t const row_base =
            axis_0 * 8192u +
            axis_1 * 512u;

         constexpr std::size_t itemsPerThread = 2u;
         T vals[itemsPerThread];
         std::size_t idxs[itemsPerThread];
         T local_sum = static_cast<T>(0);
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 256u;
            if (j < 512u) {
               std::size_t const norm_offset = j * 1u;
               std::size_t const s_norm_offset = j * 1u;
               std::size_t const b_norm_offset = j * 1u;
               idxs[k] = row_base + norm_offset;
               vals[k] = X[idxs[k]];
               local_sum += vals[k];
            } else {
               idxs[k] = 0u;
               vals[k] = static_cast<T>(0);
            }
         }

         // Pass 1: compute mean via shared-memory tree reduction
         shmem[tid] = local_sum;
         alpaka::syncBlockThreads(acc);
         if (tid < 128u) shmem[tid] += shmem[tid + 128u];
         alpaka::syncBlockThreads(acc);
         if (tid < 64u) shmem[tid] += shmem[tid + 64u];
         alpaka::syncBlockThreads(acc);
         if (tid < 32u) shmem[tid] += shmem[tid + 32u];
         alpaka::syncBlockThreads(acc);
         if (tid < 16u) shmem[tid] += shmem[tid + 16u];
         alpaka::syncBlockThreads(acc);
         if (tid < 8u) shmem[tid] += shmem[tid + 8u];
         alpaka::syncBlockThreads(acc);
         if (tid < 4u) shmem[tid] += shmem[tid + 4u];
         alpaka::syncBlockThreads(acc);
         if (tid < 2u) shmem[tid] += shmem[tid + 2u];
         alpaka::syncBlockThreads(acc);
         if (tid < 1u) shmem[tid] += shmem[tid + 1u];
         alpaka::syncBlockThreads(acc);
         T const mean = shmem[0] / static_cast<T>(512);
         alpaka::syncBlockThreads(acc);

         // Pass 2: compute variance
         T local_sq = static_cast<T>(0);
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 256u;
            if (j < 512u) { T const diff = vals[k] - mean; local_sq += diff * diff; }
         }
         shmem[tid] = local_sq;
         alpaka::syncBlockThreads(acc);
         if (tid < 128u) shmem[tid] += shmem[tid + 128u];
         alpaka::syncBlockThreads(acc);
         if (tid < 64u) shmem[tid] += shmem[tid + 64u];
         alpaka::syncBlockThreads(acc);
         if (tid < 32u) shmem[tid] += shmem[tid + 32u];
         alpaka::syncBlockThreads(acc);
         if (tid < 16u) shmem[tid] += shmem[tid + 16u];
         alpaka::syncBlockThreads(acc);
         if (tid < 8u) shmem[tid] += shmem[tid + 8u];
         alpaka::syncBlockThreads(acc);
         if (tid < 4u) shmem[tid] += shmem[tid + 4u];
         alpaka::syncBlockThreads(acc);
         if (tid < 2u) shmem[tid] += shmem[tid + 2u];
         alpaka::syncBlockThreads(acc);
         if (tid < 1u) shmem[tid] += shmem[tid + 1u];
         alpaka::syncBlockThreads(acc);
         T const invStdDev = static_cast<T>(1) / sqrt(acc, shmem[0] / static_cast<T>(512) + static_cast<T>(0.000010));


         // Pass 3: normalize + scale + bias
         std::size_t const scale_base =
            0u;
         std::size_t const bias_base =
            0u;
         for (std::size_t k = 0; k < itemsPerThread; ++k) {
            std::size_t const j = tid + k * 256u;
            if (j < 512u) {
               std::size_t const norm_offset = j * 1u;
               std::size_t const s_norm_offset = j * 1u;
               std::size_t const b_norm_offset = j * 1u;
               T out_val = scale[scale_base + s_norm_offset] * invStdDev * (vals[k] - mean);
               out_val += bias[bias_base + b_norm_offset];
               Y[idxs[k]] = out_val;
            }
         }
      }
   };

//------ TRANSPOSE_KERNEL_ALPAKA
   struct TransposeKernel_op_12 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output,const std::size_t totalElements) const {
         auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
         if (idx >= totalElements) return;
         std::size_t input_idx = 0;
         std::size_t remaining = idx;
         std::size_t coord;
            coord = remaining / (262144);
            remaining = remaining - coord * (262144);
            input_idx += coord * (128);
            coord = remaining / (128);
            remaining = remaining - coord * (128);
            input_idx += coord * (2048);
            coord = remaining / (1);
            remaining = remaining - coord * (1);
            input_idx += coord * (1);
         output[idx] = input[input_idx];
      }
   };

//------ FUSED_ELTWISE_KERNEL_14_15_16
struct FusedEltwiseKernel_14_15_16 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[((((idx) / 4194304) % 3) * 128 + (((idx) / 128) % 32768) * 384 + (((idx) / 128) % 1) * 384 + (((idx) / 128) % 1) * 12582912 + (((idx) / 1) % 128) * 1)];
         auto v_op_14_1 = v_input_0;
         auto v_op_15_2 = v_op_14_1;
         auto v_op_16_3 = v_op_15_2;
         out0[idx] = v_op_16_3;
      }
   }
};

//------ GATHER_KERNEL_ALPAKA
   struct GatherKernel_op_18 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ input,
         int64_t const* __restrict__ indices,
         T* __restrict__ output,
         std::size_t const totalElements) const {

         auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
         if (global_thread_idx >= totalElements) return;
         auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

         for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {

            std::size_t const out_0 = (elem_idx / (128)) % (32768);
            std::size_t const out_1 = (elem_idx / (128)) % (1);
            std::size_t const out_2 = (elem_idx / (1)) % (128);

            std::size_t const i_index = 0u;

            int64_t k = indices[i_index];
            if (k < 0) k += (3);
            if (k < 0) k = 0;
            if (k >= static_cast<int64_t>(3)) k = static_cast<int64_t>(3) - 1;

            std::size_t const input_idx =
               static_cast<std::size_t>(k) * (4194304) +
               out_0 * (128) +
               out_1 * (128) +
               out_2 * (1);

            output[elem_idx] = input[input_idx];
         }
      }
   };

//------ GATHER_KERNEL_ALPAKA
   struct GatherKernel_op_19 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ input,
         int64_t const* __restrict__ indices,
         T* __restrict__ output,
         std::size_t const totalElements) const {

         auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
         if (global_thread_idx >= totalElements) return;
         auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

         for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {

            std::size_t const out_0 = (elem_idx / (128)) % (32768);
            std::size_t const out_1 = (elem_idx / (128)) % (1);
            std::size_t const out_2 = (elem_idx / (1)) % (128);

            std::size_t const i_index = 0u;

            int64_t k = indices[i_index];
            if (k < 0) k += (3);
            if (k < 0) k = 0;
            if (k >= static_cast<int64_t>(3)) k = static_cast<int64_t>(3) - 1;

            std::size_t const input_idx =
               static_cast<std::size_t>(k) * (4194304) +
               out_0 * (128) +
               out_1 * (128) +
               out_2 * (1);

            output[elem_idx] = input[input_idx];
         }
      }
   };

//------ GATHER_KERNEL_ALPAKA
   struct GatherKernel_op_20 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ input,
         int64_t const* __restrict__ indices,
         T* __restrict__ output,
         std::size_t const totalElements) const {

         auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
         if (global_thread_idx >= totalElements) return;
         auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

         for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {

            std::size_t const out_0 = (elem_idx / (128)) % (32768);
            std::size_t const out_1 = (elem_idx / (128)) % (1);
            std::size_t const out_2 = (elem_idx / (1)) % (128);

            std::size_t const i_index = 0u;

            int64_t k = indices[i_index];
            if (k < 0) k += (3);
            if (k < 0) k = 0;
            if (k >= static_cast<int64_t>(3)) k = static_cast<int64_t>(3) - 1;

            std::size_t const input_idx =
               static_cast<std::size_t>(k) * (4194304) +
               out_0 * (128) +
               out_1 * (128) +
               out_2 * (1);

            output[elem_idx] = input[input_idx];
         }
      }
   };

//------ FUSED_ELTWISE_KERNEL_24_25_29
struct FusedEltwiseKernel_24_25_29 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[((((idx) / 256) % 16384) * 16 + (((idx) / 16) % 16) * 262144 + (((idx) / 1) % 16) * 1)];
         auto v_op_24_1 = v_input_0;
         auto v_op_25_2 = v_op_24_1;
         auto v_input_3 = input1[0];
         auto v_op_29_4 = v_op_25_2 * v_input_3;
         out0[idx] = v_op_29_4;
      }
   }
};

//------ FUSED_ELTWISE_KERNEL_26_30
struct FusedEltwiseKernel_26_30 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[((((idx) / 256) % 16384) * 16 + (((idx) / 16) % 16) * 1 + (((idx) / 1) % 16) * 262144)];
         auto v_op_26_1 = v_input_0;
         auto v_op_30_2 = v_op_26_1;
         out0[idx] = v_op_30_2;
      }
   }
};

//------ FUSED_ELTWISE_KERNEL_27_28
struct FusedEltwiseKernel_27_28 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[((((idx) / 256) % 16384) * 16 + (((idx) / 16) % 16) * 262144 + (((idx) / 1) % 16) * 1)];
         auto v_op_27_1 = v_input_0;
         auto v_op_28_2 = v_op_27_1;
         out0[idx] = v_op_28_2;
      }
   }
};

//------ SOFTMAX_KERNEL_ALPAKA
   struct SoftmaxKernel_op_32 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ X,
         T* __restrict__ Y,
         std::size_t const numRows) const {

         auto& smax = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
         auto& ssum = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
         auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
         auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
         if (row >= numRows) return;

         std::size_t const axis_size = 16;
         std::size_t const inner_stride = 1;
         std::size_t const row_block = axis_size * inner_stride;
         std::size_t const row_base = (row / inner_stride) * row_block + (row % inner_stride);

         // fused pass: running (max, sum) per thread
         T m = X[row_base];
         T d = static_cast<T>(0);
         for (std::size_t l = tid; l < axis_size; l += 32u) {
            T x = X[row_base + l * inner_stride];
            T m_new = (x > m) ? x : m;
            d = d * exp(acc, m - m_new) + exp(acc, x - m_new);
            m = m_new;
         }
         smax[tid] = m;
         ssum[tid] = d;
         alpaka::syncBlockThreads(acc);

         // combined (max, sum) tree reduction
         for (std::size_t s = 32u / 2u; s >= 1u; s /= 2u) {
            if (tid < s) {
               T m_a = smax[tid];
               T m_b = smax[tid + s];
               T m_r = (m_b > m_a) ? m_b : m_a;
               ssum[tid] = ssum[tid] * exp(acc, m_a - m_r) + ssum[tid + s] * exp(acc, m_b - m_r);
               smax[tid] = m_r;
            }
            alpaka::syncBlockThreads(acc);
         }
         T const vmax = smax[0];
         T const sum = ssum[0];
         alpaka::syncBlockThreads(acc);

         // normalize pass
         T const inv = static_cast<T>(1) / sum;
         for (std::size_t l = tid; l < axis_size; l += 32u) {
            std::size_t const idx = row_base + l * inner_stride;
            T e = exp(acc, X[idx] - vmax) * inv;
            Y[idx] = e;
         }
      }
   };

//------ TRANSPOSE_KERNEL_ALPAKA
   struct TransposeKernel_op_34 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output,const std::size_t totalElements) const {
         auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
         if (idx >= totalElements) return;
         std::size_t input_idx = 0;
         std::size_t remaining = idx;
         std::size_t coord;
            coord = remaining / (262144);
            remaining = remaining - coord * (262144);
            input_idx += coord * (16);
            coord = remaining / (16);
            remaining = remaining - coord * (16);
            input_idx += coord * (256);
            coord = remaining / (1);
            remaining = remaining - coord * (1);
            input_idx += coord * (1);
         output[idx] = input[input_idx];
      }
   };

//------ FUSED_ELTWISE_KERNEL_38_39
struct FusedEltwiseKernel_38_39 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[((((idx) / 2048) % 2048) * 128 + (((idx) / 128) % 16) * 262144 + (((idx) / 1) % 128) * 1)];
         auto v_op_38_1 = v_input_0;
         auto v_op_39_2 = v_op_38_1;
         out0[idx] = v_op_39_2;
      }
   }
};

//------ ReduceMean_KERNEL_ALPAKA (block parallel reduction)
   struct ReduceKernel_ReduceMean_mean {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ input,
         T* __restrict__ output,
         std::size_t const reducedLength,
         std::size_t const outputLength) const {

         auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);

         auto const thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads >(acc)[0];
         auto const out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks >(acc)[0];
         if (out_idx >= outputLength) return;

         T partial = static_cast<T>(0);
         for (std::size_t r = thread_id; r < reducedLength; r += 128u) {
            std::size_t const in_idx = out_idx * reducedLength + r;
            partial += input[in_idx];
         }

         shmem[thread_id] = partial;
         alpaka::syncBlockThreads(acc);

         std::int32_t const warpSize = alpaka::warp::getSize(acc);
         for (std::size_t s = 64u; s >= static_cast<std::size_t>(warpSize); s >>= 1u) {
            if (thread_id < s) {
               shmem[thread_id] += shmem[thread_id + s];
            }
            alpaka::syncBlockThreads(acc);
         }

         if (thread_id < static_cast<std::size_t>(warpSize)) {
            T val = shmem[thread_id];
            for (std::int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
               T const other = alpaka::warp::shfl_down(acc, val, static_cast<std::uint32_t>(offset));
               val += other;
            }

            if (thread_id == 0u) {
               T result = val;
               result /= static_cast<T>(reducedLength);
               output[out_idx] = result;
            }
         }
      }
   };

//------ FUSED_REDUCTION_KERNEL_41_42_43_44_45_46_47_48
struct FusedEltwiseKernel_41_42_43_44_45_46_47_48 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TInput2, typename TInput3, typename TInput4, typename TInput5, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TInput2 const* __restrict__ input2, TInput3 const* __restrict__ input3, TInput4 const* __restrict__ input4, TInput5 const* __restrict__ input5, TOutput0* __restrict__ out0) const {
      using T = TOutput0;
      auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);
      const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      if (out_idx >= 32768u) return;
      T partial = static_cast<T>(0);
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t in_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_0 = input0[in_idx];
         auto v_input_1 = input1[(((in_idx) / 2048u) % 2048u) * 16u + (((in_idx) / 128u) % 16u)];
         auto v_op_41_2 = v_input_0 - v_input_1;
         auto v_input_3 = input2[0];
         auto v_op_42_4 = ((v_op_41_2) * (v_op_41_2));
         partial = ((partial) + (v_op_42_4));
      }
      shmem[thread_id] = partial;
      alpaka::syncBlockThreads(acc);
      for (std::size_t s = 64u; s > 0u; s >>= 1u) {
         if (thread_id < s) shmem[thread_id] = ((shmem[thread_id]) + (shmem[thread_id + s]));
         alpaka::syncBlockThreads(acc);
      }
      if (thread_id == 0u) shmem[0] = ((shmem[0]) / static_cast<T>(128));
      alpaka::syncBlockThreads(acc);
      const T reduction_value = shmem[0];
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t element_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_5 = input4[0];
         auto v_input_6 = input0[element_idx];
         auto v_input_7 = input1[(((element_idx) / 2048u) % 2048u) * 16u + (((element_idx) / 128u) % 16u)];
         auto v_op_41_8 = v_input_6 - v_input_7;
         auto v_op_46_9 = v_input_5 * v_op_41_8;
         auto v_input_10 = input3[0];
         auto v_op_44_11 = reduction_value * v_input_10;
         auto v_op_45_12 = std::sqrt(v_op_44_11);
         auto v_input_13 = input5[0];
         auto v_op_47_14 = v_op_45_12 + v_input_13;
         auto v_op_48_15 = v_op_46_9 / v_op_47_14;
         out0[element_idx] = v_op_48_15;
      }
   }
};

//------ FUSED_REDUCTION_KERNEL_49_50_51_52
struct FusedEltwiseKernel_49_50_51_52 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TInput2, typename TOutput0, typename TOutput1>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TInput2 const* __restrict__ input2, TOutput0* __restrict__ out0, TOutput1* __restrict__ out1) const {
      using T = TOutput0;
      auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);
      const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      if (out_idx >= 32768u) return;
      T partial = static_cast<T>(0);
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t in_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_0 = input2[in_idx];
         auto v_input_1 = input0[in_idx];
         auto v_input_2 = input1[0];
         auto v_op_49_3 = v_input_1 + v_input_2;
         auto v_op_50_4 = v_input_0 + v_op_49_3;
         partial = ((partial) + (v_op_50_4));
      }
      shmem[thread_id] = partial;
      alpaka::syncBlockThreads(acc);
      for (std::size_t s = 64u; s > 0u; s >>= 1u) {
         if (thread_id < s) shmem[thread_id] = ((shmem[thread_id]) + (shmem[thread_id + s]));
         alpaka::syncBlockThreads(acc);
      }
      if (thread_id == 0u) shmem[0] = ((shmem[0]) / static_cast<T>(128));
      alpaka::syncBlockThreads(acc);
      const T reduction_value = shmem[0];
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t element_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_5 = input2[element_idx];
         auto v_input_6 = input0[element_idx];
         auto v_input_7 = input1[0];
         auto v_op_49_8 = v_input_6 + v_input_7;
         auto v_op_50_9 = v_input_5 + v_op_49_8;
         out0[element_idx] = v_op_50_9;
      }
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t element_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_10 = input2[element_idx];
         auto v_input_11 = input0[element_idx];
         auto v_input_12 = input1[0];
         auto v_op_49_13 = v_input_11 + v_input_12;
         auto v_op_50_14 = v_input_10 + v_op_49_13;
         auto v_op_52_15 = v_op_50_14 - reduction_value;
         out1[element_idx] = v_op_52_15;
      }
   }
};

//------ FUSED_REDUCTION_KERNEL_53_54_55_56_57_58_59_60
struct FusedEltwiseKernel_53_54_55_56_57_58_59_60 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TInput2, typename TInput3, typename TInput4, typename TInput5, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TInput2 const* __restrict__ input2, TInput3 const* __restrict__ input3, TInput4 const* __restrict__ input4, TInput5 const* __restrict__ input5, TOutput0* __restrict__ out0) const {
      using T = TOutput0;
      auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);
      const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      if (out_idx >= 32768u) return;
      T partial = static_cast<T>(0);
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t in_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_0 = input0[in_idx];
         auto v_input_1 = input1[0];
         auto v_op_53_2 = ((v_input_0) * (v_input_0));
         partial = ((partial) + (v_op_53_2));
      }
      shmem[thread_id] = partial;
      alpaka::syncBlockThreads(acc);
      for (std::size_t s = 64u; s > 0u; s >>= 1u) {
         if (thread_id < s) shmem[thread_id] = ((shmem[thread_id]) + (shmem[thread_id + s]));
         alpaka::syncBlockThreads(acc);
      }
      if (thread_id == 0u) shmem[0] = ((shmem[0]) / static_cast<T>(128));
      alpaka::syncBlockThreads(acc);
      const T reduction_value = shmem[0];
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t element_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_3 = input3[0];
         auto v_input_4 = input0[element_idx];
         auto v_op_57_5 = v_input_3 * v_input_4;
         auto v_input_6 = input2[0];
         auto v_op_55_7 = reduction_value * v_input_6;
         auto v_op_56_8 = std::sqrt(v_op_55_7);
         auto v_input_9 = input4[0];
         auto v_op_58_10 = v_op_56_8 + v_input_9;
         auto v_op_59_11 = v_op_57_5 / v_op_58_10;
         auto v_input_12 = input5[0];
         auto v_op_60_13 = v_op_59_11 + v_input_12;
         out0[element_idx] = v_op_60_13;
      }
   }
};

//------ FUSED_REDUCTION_KERNEL_64_65_66
struct FusedEltwiseKernel_64_65_66 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TOutput0* __restrict__ out0) const {
      using T = TOutput0;
      auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);
      const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      if (out_idx >= 32768u) return;
      T partial = static_cast<T>(0);
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t in_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_0 = input0[in_idx];
         auto v_input_1 = input1[in_idx];
         auto v_op_64_2 = v_input_0 + v_input_1;
         partial = ((partial) + (v_op_64_2));
      }
      shmem[thread_id] = partial;
      alpaka::syncBlockThreads(acc);
      for (std::size_t s = 64u; s > 0u; s >>= 1u) {
         if (thread_id < s) shmem[thread_id] = ((shmem[thread_id]) + (shmem[thread_id + s]));
         alpaka::syncBlockThreads(acc);
      }
      if (thread_id == 0u) shmem[0] = ((shmem[0]) / static_cast<T>(128));
      alpaka::syncBlockThreads(acc);
      const T reduction_value = shmem[0];
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t element_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_3 = input0[element_idx];
         auto v_input_4 = input1[element_idx];
         auto v_op_64_5 = v_input_3 + v_input_4;
         auto v_op_66_6 = v_op_64_5 - reduction_value;
         out0[element_idx] = v_op_66_6;
      }
   }
};

//------ FUSED_REDUCTION_KERNEL_67_68_69_70_71_72_73_74
struct FusedEltwiseKernel_67_68_69_70_71_72_73_74 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TInput2, typename TInput3, typename TInput4, typename TInput5, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TInput2 const* __restrict__ input2, TInput3 const* __restrict__ input3, TInput4 const* __restrict__ input4, TInput5 const* __restrict__ input5, TOutput0* __restrict__ out0) const {
      using T = TOutput0;
      auto& shmem = alpaka::declareSharedVar<T[128], __COUNTER__>(acc);
      const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      if (out_idx >= 32768u) return;
      T partial = static_cast<T>(0);
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t in_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_0 = input0[in_idx];
         auto v_input_1 = input1[0];
         auto v_op_67_2 = ((v_input_0) * (v_input_0));
         partial = ((partial) + (v_op_67_2));
      }
      shmem[thread_id] = partial;
      alpaka::syncBlockThreads(acc);
      for (std::size_t s = 64u; s > 0u; s >>= 1u) {
         if (thread_id < s) shmem[thread_id] = ((shmem[thread_id]) + (shmem[thread_id + s]));
         alpaka::syncBlockThreads(acc);
      }
      if (thread_id == 0u) shmem[0] = ((shmem[0]) / static_cast<T>(128));
      alpaka::syncBlockThreads(acc);
      const T reduction_value = shmem[0];
      for (std::size_t r = thread_id; r < 128u; r += 128u) {
         const std::size_t element_idx = ((out_idx / 16u) % 2048u) * 2048u + ((out_idx / 1u) % 16u) * 128u + ((r / 1u) % 128u);
         auto v_input_3 = input3[0];
         auto v_input_4 = input0[element_idx];
         auto v_op_71_5 = v_input_3 * v_input_4;
         auto v_input_6 = input2[0];
         auto v_op_69_7 = reduction_value * v_input_6;
         auto v_op_70_8 = std::sqrt(v_op_69_7);
         auto v_input_9 = input4[0];
         auto v_op_72_10 = v_op_70_8 + v_input_9;
         auto v_op_73_11 = v_op_71_5 / v_op_72_10;
         auto v_input_12 = input5[0];
         auto v_op_74_13 = v_op_73_11 + v_input_12;
         out0[element_idx] = v_op_74_13;
      }
   }
};

//------ TANH_KERNEL_ALPAKA
struct TanhKernel {
   template<typename TAcc, typename T>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < numElements) { out[idx] = tanh(data[idx]); }
   }
};

//------ SOFTMAX_KERNEL_ALPAKA
   struct SoftmaxKernel_op_78 {
      template<typename TAcc, typename T>
      ALPAKA_FN_ACC void operator()(
         TAcc const& acc,
         T const* __restrict__ X,
         T* __restrict__ Y,
         std::size_t const numRows) const {

         auto& smax = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
         auto& ssum = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
         auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
         auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
         if (row >= numRows) return;

         std::size_t const axis_size = 16;
         std::size_t const inner_stride = 1;
         std::size_t const row_block = axis_size * inner_stride;
         std::size_t const row_base = (row / inner_stride) * row_block + (row % inner_stride);

         // fused pass: running (max, sum) per thread
         T m = X[row_base];
         T d = static_cast<T>(0);
         for (std::size_t l = tid; l < axis_size; l += 32u) {
            T x = X[row_base + l * inner_stride];
            T m_new = (x > m) ? x : m;
            d = d * exp(acc, m - m_new) + exp(acc, x - m_new);
            m = m_new;
         }
         smax[tid] = m;
         ssum[tid] = d;
         alpaka::syncBlockThreads(acc);

         // combined (max, sum) tree reduction
         for (std::size_t s = 32u / 2u; s >= 1u; s /= 2u) {
            if (tid < s) {
               T m_a = smax[tid];
               T m_b = smax[tid + s];
               T m_r = (m_b > m_a) ? m_b : m_a;
               ssum[tid] = ssum[tid] * exp(acc, m_a - m_r) + ssum[tid + s] * exp(acc, m_b - m_r);
               smax[tid] = m_r;
            }
            alpaka::syncBlockThreads(acc);
         }
         T const vmax = smax[0];
         T const sum = ssum[0];
         alpaka::syncBlockThreads(acc);

         // normalize pass
         T const inv = static_cast<T>(1) / sum;
         for (std::size_t l = tid; l < axis_size; l += 32u) {
            std::size_t const idx = row_base + l * inner_stride;
            T e = exp(acc, X[idx] - vmax) * inv;
            Y[idx] = e;
         }
      }
   };

//------ FUSED_REDUCTION_KERNEL_79_80
struct FusedEltwiseKernel_79_80 {
   template<typename TAcc, typename TInput0, typename TInput1, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TInput1 const* __restrict__ input1, TOutput0* __restrict__ out0) const {
      using T = TOutput0;
      auto& shmem = alpaka::declareSharedVar<T[32], __COUNTER__>(acc);
      const auto out_idx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
      const auto thread_id = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
      if (out_idx >= 262144u) return;
      T partial = static_cast<T>(0);
      for (std::size_t r = thread_id; r < 16u; r += 32u) {
         const std::size_t in_idx = ((out_idx / 128u) % 2048u) * 2048u + ((r / 1u) % 16u) * 128u + ((out_idx / 1u) % 128u);
         auto v_input_0 = input0[(((in_idx) / 2048u) % 2048u) * 16u + (((in_idx) / 128u) % 16u)];
         auto v_input_1 = input1[in_idx];
         auto v_op_79_2 = v_input_0 * v_input_1;
         partial = ((partial) + (v_op_79_2));
      }
      shmem[thread_id] = partial;
      alpaka::syncBlockThreads(acc);
      for (std::size_t s = 16u; s > 0u; s >>= 1u) {
         if (thread_id < s) shmem[thread_id] = ((shmem[thread_id]) + (shmem[thread_id + s]));
         alpaka::syncBlockThreads(acc);
      }
      if (thread_id == 0u) shmem[0] = shmem[0];
      alpaka::syncBlockThreads(acc);
      const T reduction_value = shmem[0];
      if (thread_id == 0u) {
         out0[out_idx] = reduction_value;
      }
   }
};

//------ FUSED_ELTWISE_KERNEL_84_89
struct FusedEltwiseKernel_84_89 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[idx];
         auto v_op_84_1 = v_input_0;
         auto v_op_89_2 = v_op_84_1;
         out0[idx] = v_op_89_2;
      }
   }
};

//------ FUSED_ELTWISE_KERNEL_86_91
struct FusedEltwiseKernel_86_91 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[idx];
         auto v_op_86_1 = v_input_0;
         auto v_op_91_2 = v_op_86_1;
         out0[idx] = v_op_91_2;
      }
   }
};

//------ FUSED_ELTWISE_KERNEL_88_90
struct FusedEltwiseKernel_88_90 {
   template<typename TAcc, typename TInput0, typename TOutput0>
   ALPAKA_FN_ACC void operator()(TAcc const& acc, TInput0 const* __restrict__ input0, TOutput0* __restrict__ out0, std::size_t n) const {
      using T = TOutput0;
      const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (idx < n) {
         auto v_input_0 = input0[idx];
         auto v_op_88_1 = v_input_0;
         auto v_op_90_2 = v_op_88_1;
         out0[idx] = v_op_90_2;
      }
   }
};

template <typename tagAcc>
struct Session {

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Acc = alpaka::TagToAcc<tagAcc, Dim, Idx>;
using DevAcc = alpaka::Dev<Acc>;

using QueueProperty = std::conditional_t< // CPU wants Blocking queue
    std::is_same_v<tagAcc, alpaka::TagCpuSerial>,
    alpaka::Blocking,
    alpaka::NonBlocking>;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

using BufF1D = alpaka::Buf<Acc, float, Dim, Idx>;
using BufD1D = alpaka::Buf<Acc, double, Dim, Idx>;
using BufI321D = alpaka::Buf<Acc, int32_t, Dim, Idx>;
using BufI641D = alpaka::Buf<Acc, int64_t, Dim, Idx>;
using BufUI81D = alpaka::Buf<Acc, uint8_t, Dim, Idx>;

// Non-owning device view types (ViewPlainPtr) for the span-based infer interface
using ViewF1D = alpaka::ViewPlainPtr<DevAcc, float, Dim, Idx>;
using ViewConstF1D = alpaka::ViewPlainPtr<DevAcc, const float, Dim, Idx>;
using ViewD1D = alpaka::ViewPlainPtr<DevAcc, double, Dim, Idx>;
using ViewConstD1D = alpaka::ViewPlainPtr<DevAcc, const double, Dim, Idx>;
using ViewI321D = alpaka::ViewPlainPtr<DevAcc, int32_t, Dim, Idx>;
using ViewConstI321D = alpaka::ViewPlainPtr<DevAcc, const int32_t, Dim, Idx>;
using ViewI641D = alpaka::ViewPlainPtr<DevAcc, int64_t, Dim, Idx>;
using ViewConstI641D = alpaka::ViewPlainPtr<DevAcc, const int64_t, Dim, Idx>;
using ViewUI81D = alpaka::ViewPlainPtr<DevAcc, uint8_t, Dim, Idx>;
using ViewConstUI81D = alpaka::ViewPlainPtr<DevAcc, const uint8_t, Dim, Idx>;


alpaka::Platform<Acc> const platform{};
DevAcc devAcc = alpaka::getDevByIdx(platform, 0);
alpaka::PlatformCpu platformHost{};
alpaka::DevCpu hostAcc = alpaka::getDevByIdx(platformHost, 0);
QueueAcc queue{devAcc};
Idx threadsPerBlock = 256;

using Ext1D = alpaka::Vec<Dim, Idx>;
using Vec = alpaka::Vec<Dim, Idx>;


// BLAS declarations
sofieBLAS<tagAcc> blas{queue};

// initialized tensors for weights
int64_t tensor_val_32[3] = { 16, 16384, 16};
BufI641D deviceBuf_val_32 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{3}));
int64_t tensor_val_30[1] = { 16384};
BufI641D deviceBuf_val_30 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelprojection_layerlinear_classifierweight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{32}));
int64_t tensor_val_2[1] = { 1};
BufI641D deviceBuf_val_2 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_52[3] = { 16, 2048, 128};
BufI641D deviceBuf_val_52 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{3}));
BufF1D deviceBuf_val_43 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_sym_size_int_2[1] = { 2048};
BufI641D deviceBuf_sym_size_int_2 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelprojection_layerlinear1bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{64}));
BufF1D deviceBuf_modelprojection_layerlinear1weight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{8192}));
BufF1D deviceBuf_modelparticle_embedembed6weight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{512}));
BufF1D deviceBuf_modelencodernormalpha = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_9 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{65536}));
BufF1D deviceBuf_modelencoderlayers0residual_connections1normbias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelparticle_embedembed3weight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{128}));
BufF1D deviceBuf_modelparticle_embedembed0weight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{10}));
BufF1D deviceBuf_modelprojection_layerpooling_layerscore_layer2bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelparticle_embedembed0bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{10}));
BufF1D deviceBuf_modelparticle_embedembed1bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{128}));
BufF1D deviceBuf_modelencodernormbias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelprojection_layerlinear_chargeweight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{32}));
BufF1D deviceBuf_modelprojection_layerlinear2bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{32}));
BufF1D deviceBuf_modelparticle_embedembed3bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{128}));
BufF1D deviceBuf_modelparticle_embedembed4bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{512}));
BufF1D deviceBuf_modelprojection_layerlinear2weight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{2048}));
BufF1D deviceBuf_modelencoderlayers0residual_connections1normalpha = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelencoderlayers0self_attention_blockin_proj_bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{384}));
BufF1D deviceBuf_modelencoderlayers0residual_connections0normalpha = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_75 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{8192}));
BufF1D deviceBuf_modelencoderlayers0self_attention_blockout_projbias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{128}));
BufF1D deviceBuf_modelprojection_layerlinear_regressor_ptweight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{32}));
BufF1D deviceBuf_val_66 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{65536}));
int64_t tensor_val_81[2] = { -1, 128};
BufI641D deviceBuf_val_81 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{2}));
BufF1D deviceBuf_val_56 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelparticle_embedembed6bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{512}));
BufF1D deviceBuf_modelencoderlayers0feed_forward_blocklinear_1bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{512}));
BufF1D deviceBuf_modelprojection_layerlinear_classifierbias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_25[1] = { 0};
BufI641D deviceBuf_val_25 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelencoderlayers0residual_connections0normbias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelprojection_layerlinear_chargebias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_5 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1280}));
BufF1D deviceBuf_modelprojection_layerlinear_regressor_ptbias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_13 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{65536}));
BufF1D deviceBuf_val_15 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{49152}));
int64_t tensor_val_55[1] = { -1};
BufI641D deviceBuf_val_55 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_77 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{64}));
int64_t tensor_mul_55[1] = { 16384};
BufI641D deviceBuf_mul_55 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_80[4] = { -1, 1, 3, 128};
BufI641D deviceBuf_val_80 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{4}));
BufF1D deviceBuf_modelencoderlayers0self_attention_blockout_projweight = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{16384}));
BufF1D deviceBuf_modelparticle_embedembed7bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{128}));
int64_t tensor_val_23[1] = { 0};
BufI641D deviceBuf_val_23 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_31[1] = { 16};
BufI641D deviceBuf_val_31 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_24[1] = { -2};
BufI641D deviceBuf_val_24 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_26[1] = { 2};
BufI641D deviceBuf_val_26 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelprojection_layerpooling_layerscore_layer0bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{64}));
int64_t tensor_val_27[1] = { 8};
BufI641D deviceBuf_val_27 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_51[1] = { 128};
BufI641D deviceBuf_val_51 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_59 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_60 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_modelencoderlayers0feed_forward_blocklinear_2bias = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{128}));
int64_t tensor_val_0[1] = { 2048};
BufI641D deviceBuf_val_0 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
BufF1D deviceBuf_val_68 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{65536}));
int64_t tensor_val_79[1] = { 1};
BufI641D deviceBuf_val_79 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));
int64_t tensor_val_1[1] = { 16};
BufI641D deviceBuf_val_1 = alpaka::allocBuf<int64_t, Idx>(devAcc, Ext1D::all(Idx{1}));

//--- declare intermediate tensors
ViewF1D deviceBuf_reg_pt{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_class_logit{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_12{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_sum_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_8{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_7{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_softmax_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_sub_100{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_transpose_4{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_relu_4{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_select{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_relu_5{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_select_2{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_transpose_5{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_select_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_relu{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_13{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_transpose_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_transpose_6{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_layer_norm{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
BufF1D deviceBuf_linear_1 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{16777216}));
ViewF1D deviceBuf_softmax{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_mul_74{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
BufF1D deviceBuf_linear = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{4194304}));
BufF1D deviceBuf_linear_5 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{16777216}));
ViewF1D deviceBuf_charge_logit{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_transpose{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_bmm{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
BufF1D deviceBuf_linear_2 = alpaka::allocBuf<float, Idx>(devAcc, Ext1D::all(Idx{4194304}));
ViewF1D deviceBuf_relu_3{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_relu_2{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_11{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_6{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_add_278{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_layer_norm_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_tanh{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_relu_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_div{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_layer_norm_2{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_4{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_mean{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_transpose_7{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_linear_3{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_bmm_1{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_add_189{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_sub_81{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};
ViewF1D deviceBuf_add_224{static_cast<float*>(nullptr), devAcc, Ext1D::all(Idx{0})};

//--- unified memory pool buffer (allocated in constructor)
BufUI81D fUnifiedPool = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{1}));
std::size_t fUnifiedPoolSize = 0;
std::size_t GetIntermediateMemoryPoolSize() const { return fUnifiedPoolSize; }
std::size_t GetLargestFreeBlock() const { return 0; }
std::size_t GetTotalFreeMemory() const { return 0; }
double GetFragmentation() const { return 0.0; }

void InitSession(std::string filename) {
// temporary initialized tensors for loading weights
std::vector<float> tensor_modelprojection_layerlinear_classifierweight(32);
std::vector<float> tensor_val_43(1);
std::vector<float> tensor_modelprojection_layerlinear1bias(64);
std::vector<float> tensor_modelprojection_layerlinear1weight(8192);
std::vector<float> tensor_modelparticle_embedembed6weight(512);
std::vector<float> tensor_modelencodernormalpha(1);
std::vector<float> tensor_val_9(65536);
std::vector<float> tensor_modelencoderlayers0residual_connections1normbias(1);
std::vector<float> tensor_modelparticle_embedembed3weight(128);
std::vector<float> tensor_modelparticle_embedembed0weight(10);
std::vector<float> tensor_modelprojection_layerpooling_layerscore_layer2bias(1);
std::vector<float> tensor_modelparticle_embedembed0bias(10);
std::vector<float> tensor_modelparticle_embedembed1bias(128);
std::vector<float> tensor_modelencodernormbias(1);
std::vector<float> tensor_modelprojection_layerlinear_chargeweight(32);
std::vector<float> tensor_modelprojection_layerlinear2bias(32);
std::vector<float> tensor_modelparticle_embedembed3bias(128);
std::vector<float> tensor_modelparticle_embedembed4bias(512);
std::vector<float> tensor_modelprojection_layerlinear2weight(2048);
std::vector<float> tensor_modelencoderlayers0residual_connections1normalpha(1);
std::vector<float> tensor_modelencoderlayers0self_attention_blockin_proj_bias(384);
std::vector<float> tensor_modelencoderlayers0residual_connections0normalpha(1);
std::vector<float> tensor_val_75(8192);
std::vector<float> tensor_modelencoderlayers0self_attention_blockout_projbias(128);
std::vector<float> tensor_modelprojection_layerlinear_regressor_ptweight(32);
std::vector<float> tensor_val_66(65536);
std::vector<float> tensor_val_56(1);
std::vector<float> tensor_modelparticle_embedembed6bias(512);
std::vector<float> tensor_modelencoderlayers0feed_forward_blocklinear_1bias(512);
std::vector<float> tensor_modelprojection_layerlinear_classifierbias(1);
std::vector<float> tensor_modelencoderlayers0residual_connections0normbias(1);
std::vector<float> tensor_modelprojection_layerlinear_chargebias(1);
std::vector<float> tensor_val_5(1280);
std::vector<float> tensor_modelprojection_layerlinear_regressor_ptbias(1);
std::vector<float> tensor_val_13(65536);
std::vector<float> tensor_val_15(49152);
std::vector<float> tensor_val_77(64);
std::vector<float> tensor_modelencoderlayers0self_attention_blockout_projweight(16384);
std::vector<float> tensor_modelparticle_embedembed7bias(128);
std::vector<float> tensor_modelprojection_layerpooling_layerscore_layer0bias(64);
std::vector<float> tensor_val_59(1);
std::vector<float> tensor_val_60(1);
std::vector<float> tensor_modelencoderlayers0feed_forward_blocklinear_2bias(128);
std::vector<float> tensor_val_68(65536);

//--- reading weights from file
using SOFIE::ReadTensorFromStream;
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("sofie failed to open file " + filename + " for input weights");
   }
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear_classifierweight, "tensor_modelprojection_layerlinear_classifierweight", 32);
   ReadTensorFromStream(f, tensor_val_43, "tensor_val_43", 1);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear1bias, "tensor_modelprojection_layerlinear1bias", 64);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear1weight, "tensor_modelprojection_layerlinear1weight", 8192);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed6weight, "tensor_modelparticle_embedembed6weight", 512);
   ReadTensorFromStream(f, tensor_modelencodernormalpha, "tensor_modelencodernormalpha", 1);
   ReadTensorFromStream(f, tensor_val_9, "tensor_val_9", 65536);
   ReadTensorFromStream(f, tensor_modelencoderlayers0residual_connections1normbias, "tensor_modelencoderlayers0residual_connections1normbias", 1);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed3weight, "tensor_modelparticle_embedembed3weight", 128);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed0weight, "tensor_modelparticle_embedembed0weight", 10);
   ReadTensorFromStream(f, tensor_modelprojection_layerpooling_layerscore_layer2bias, "tensor_modelprojection_layerpooling_layerscore_layer2bias", 1);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed0bias, "tensor_modelparticle_embedembed0bias", 10);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed1bias, "tensor_modelparticle_embedembed1bias", 128);
   ReadTensorFromStream(f, tensor_modelencodernormbias, "tensor_modelencodernormbias", 1);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear_chargeweight, "tensor_modelprojection_layerlinear_chargeweight", 32);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear2bias, "tensor_modelprojection_layerlinear2bias", 32);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed3bias, "tensor_modelparticle_embedembed3bias", 128);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed4bias, "tensor_modelparticle_embedembed4bias", 512);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear2weight, "tensor_modelprojection_layerlinear2weight", 2048);
   ReadTensorFromStream(f, tensor_modelencoderlayers0residual_connections1normalpha, "tensor_modelencoderlayers0residual_connections1normalpha", 1);
   ReadTensorFromStream(f, tensor_modelencoderlayers0self_attention_blockin_proj_bias, "tensor_modelencoderlayers0self_attention_blockin_proj_bias", 384);
   ReadTensorFromStream(f, tensor_modelencoderlayers0residual_connections0normalpha, "tensor_modelencoderlayers0residual_connections0normalpha", 1);
   ReadTensorFromStream(f, tensor_val_75, "tensor_val_75", 8192);
   ReadTensorFromStream(f, tensor_modelencoderlayers0self_attention_blockout_projbias, "tensor_modelencoderlayers0self_attention_blockout_projbias", 128);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear_regressor_ptweight, "tensor_modelprojection_layerlinear_regressor_ptweight", 32);
   ReadTensorFromStream(f, tensor_val_66, "tensor_val_66", 65536);
   ReadTensorFromStream(f, tensor_val_56, "tensor_val_56", 1);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed6bias, "tensor_modelparticle_embedembed6bias", 512);
   ReadTensorFromStream(f, tensor_modelencoderlayers0feed_forward_blocklinear_1bias, "tensor_modelencoderlayers0feed_forward_blocklinear_1bias", 512);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear_classifierbias, "tensor_modelprojection_layerlinear_classifierbias", 1);
   ReadTensorFromStream(f, tensor_modelencoderlayers0residual_connections0normbias, "tensor_modelencoderlayers0residual_connections0normbias", 1);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear_chargebias, "tensor_modelprojection_layerlinear_chargebias", 1);
   ReadTensorFromStream(f, tensor_val_5, "tensor_val_5", 1280);
   ReadTensorFromStream(f, tensor_modelprojection_layerlinear_regressor_ptbias, "tensor_modelprojection_layerlinear_regressor_ptbias", 1);
   ReadTensorFromStream(f, tensor_val_13, "tensor_val_13", 65536);
   ReadTensorFromStream(f, tensor_val_15, "tensor_val_15", 49152);
   ReadTensorFromStream(f, tensor_val_77, "tensor_val_77", 64);
   ReadTensorFromStream(f, tensor_modelencoderlayers0self_attention_blockout_projweight, "tensor_modelencoderlayers0self_attention_blockout_projweight", 16384);
   ReadTensorFromStream(f, tensor_modelparticle_embedembed7bias, "tensor_modelparticle_embedembed7bias", 128);
   ReadTensorFromStream(f, tensor_modelprojection_layerpooling_layerscore_layer0bias, "tensor_modelprojection_layerpooling_layerscore_layer0bias", 64);
   ReadTensorFromStream(f, tensor_val_59, "tensor_val_59", 1);
   ReadTensorFromStream(f, tensor_val_60, "tensor_val_60", 1);
   ReadTensorFromStream(f, tensor_modelencoderlayers0feed_forward_blocklinear_2bias, "tensor_modelencoderlayers0feed_forward_blocklinear_2bias", 128);
   ReadTensorFromStream(f, tensor_val_68, "tensor_val_68", 65536);
   f.close();

     auto hostBuf_val_32 = alpaka::createView(hostAcc, tensor_val_32, 3);
     alpaka::memcpy(queue, deviceBuf_val_32, hostBuf_val_32);
     auto hostBuf_modelprojection_layerlinear_classifierweight = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear_classifierweight, 32);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear_classifierweight, hostBuf_modelprojection_layerlinear_classifierweight);
     auto hostBuf_val_2 = alpaka::createView(hostAcc, tensor_val_2, 1);
     alpaka::memcpy(queue, deviceBuf_val_2, hostBuf_val_2);
     auto hostBuf_val_52 = alpaka::createView(hostAcc, tensor_val_52, 3);
     alpaka::memcpy(queue, deviceBuf_val_52, hostBuf_val_52);
     auto hostBuf_val_43 = alpaka::createView(hostAcc, tensor_val_43, 1);
     alpaka::memcpy(queue, deviceBuf_val_43, hostBuf_val_43);
     auto hostBuf_modelprojection_layerlinear1bias = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear1bias, 64);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear1bias, hostBuf_modelprojection_layerlinear1bias);
     auto hostBuf_modelprojection_layerlinear1weight = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear1weight, 8192);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear1weight, hostBuf_modelprojection_layerlinear1weight);
     auto hostBuf_modelparticle_embedembed6weight = alpaka::createView(hostAcc, tensor_modelparticle_embedembed6weight, 512);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed6weight, hostBuf_modelparticle_embedembed6weight);
     auto hostBuf_modelencodernormalpha = alpaka::createView(hostAcc, tensor_modelencodernormalpha, 1);
     alpaka::memcpy(queue, deviceBuf_modelencodernormalpha, hostBuf_modelencodernormalpha);
     auto hostBuf_val_9 = alpaka::createView(hostAcc, tensor_val_9, 65536);
     alpaka::memcpy(queue, deviceBuf_val_9, hostBuf_val_9);
     auto hostBuf_modelencoderlayers0residual_connections1normbias = alpaka::createView(hostAcc, tensor_modelencoderlayers0residual_connections1normbias, 1);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0residual_connections1normbias, hostBuf_modelencoderlayers0residual_connections1normbias);
     auto hostBuf_modelparticle_embedembed3weight = alpaka::createView(hostAcc, tensor_modelparticle_embedembed3weight, 128);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed3weight, hostBuf_modelparticle_embedembed3weight);
     auto hostBuf_modelparticle_embedembed0weight = alpaka::createView(hostAcc, tensor_modelparticle_embedembed0weight, 10);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed0weight, hostBuf_modelparticle_embedembed0weight);
     auto hostBuf_modelprojection_layerpooling_layerscore_layer2bias = alpaka::createView(hostAcc, tensor_modelprojection_layerpooling_layerscore_layer2bias, 1);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerpooling_layerscore_layer2bias, hostBuf_modelprojection_layerpooling_layerscore_layer2bias);
     auto hostBuf_modelparticle_embedembed0bias = alpaka::createView(hostAcc, tensor_modelparticle_embedembed0bias, 10);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed0bias, hostBuf_modelparticle_embedembed0bias);
     auto hostBuf_modelparticle_embedembed1bias = alpaka::createView(hostAcc, tensor_modelparticle_embedembed1bias, 128);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed1bias, hostBuf_modelparticle_embedembed1bias);
     auto hostBuf_modelencodernormbias = alpaka::createView(hostAcc, tensor_modelencodernormbias, 1);
     alpaka::memcpy(queue, deviceBuf_modelencodernormbias, hostBuf_modelencodernormbias);
     auto hostBuf_modelprojection_layerlinear_chargeweight = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear_chargeweight, 32);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear_chargeweight, hostBuf_modelprojection_layerlinear_chargeweight);
     auto hostBuf_modelprojection_layerlinear2bias = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear2bias, 32);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear2bias, hostBuf_modelprojection_layerlinear2bias);
     auto hostBuf_modelparticle_embedembed3bias = alpaka::createView(hostAcc, tensor_modelparticle_embedembed3bias, 128);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed3bias, hostBuf_modelparticle_embedembed3bias);
     auto hostBuf_modelparticle_embedembed4bias = alpaka::createView(hostAcc, tensor_modelparticle_embedembed4bias, 512);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed4bias, hostBuf_modelparticle_embedembed4bias);
     auto hostBuf_modelprojection_layerlinear2weight = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear2weight, 2048);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear2weight, hostBuf_modelprojection_layerlinear2weight);
     auto hostBuf_modelencoderlayers0residual_connections1normalpha = alpaka::createView(hostAcc, tensor_modelencoderlayers0residual_connections1normalpha, 1);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0residual_connections1normalpha, hostBuf_modelencoderlayers0residual_connections1normalpha);
     auto hostBuf_modelencoderlayers0self_attention_blockin_proj_bias = alpaka::createView(hostAcc, tensor_modelencoderlayers0self_attention_blockin_proj_bias, 384);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0self_attention_blockin_proj_bias, hostBuf_modelencoderlayers0self_attention_blockin_proj_bias);
     auto hostBuf_modelencoderlayers0residual_connections0normalpha = alpaka::createView(hostAcc, tensor_modelencoderlayers0residual_connections0normalpha, 1);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0residual_connections0normalpha, hostBuf_modelencoderlayers0residual_connections0normalpha);
     auto hostBuf_val_75 = alpaka::createView(hostAcc, tensor_val_75, 8192);
     alpaka::memcpy(queue, deviceBuf_val_75, hostBuf_val_75);
     auto hostBuf_modelencoderlayers0self_attention_blockout_projbias = alpaka::createView(hostAcc, tensor_modelencoderlayers0self_attention_blockout_projbias, 128);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0self_attention_blockout_projbias, hostBuf_modelencoderlayers0self_attention_blockout_projbias);
     auto hostBuf_modelprojection_layerlinear_regressor_ptweight = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear_regressor_ptweight, 32);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear_regressor_ptweight, hostBuf_modelprojection_layerlinear_regressor_ptweight);
     auto hostBuf_val_66 = alpaka::createView(hostAcc, tensor_val_66, 65536);
     alpaka::memcpy(queue, deviceBuf_val_66, hostBuf_val_66);
     auto hostBuf_val_81 = alpaka::createView(hostAcc, tensor_val_81, 2);
     alpaka::memcpy(queue, deviceBuf_val_81, hostBuf_val_81);
     auto hostBuf_val_56 = alpaka::createView(hostAcc, tensor_val_56, 1);
     alpaka::memcpy(queue, deviceBuf_val_56, hostBuf_val_56);
     auto hostBuf_modelparticle_embedembed6bias = alpaka::createView(hostAcc, tensor_modelparticle_embedembed6bias, 512);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed6bias, hostBuf_modelparticle_embedembed6bias);
     auto hostBuf_modelencoderlayers0feed_forward_blocklinear_1bias = alpaka::createView(hostAcc, tensor_modelencoderlayers0feed_forward_blocklinear_1bias, 512);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0feed_forward_blocklinear_1bias, hostBuf_modelencoderlayers0feed_forward_blocklinear_1bias);
     auto hostBuf_modelprojection_layerlinear_classifierbias = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear_classifierbias, 1);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear_classifierbias, hostBuf_modelprojection_layerlinear_classifierbias);
     auto hostBuf_val_25 = alpaka::createView(hostAcc, tensor_val_25, 1);
     alpaka::memcpy(queue, deviceBuf_val_25, hostBuf_val_25);
     auto hostBuf_modelencoderlayers0residual_connections0normbias = alpaka::createView(hostAcc, tensor_modelencoderlayers0residual_connections0normbias, 1);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0residual_connections0normbias, hostBuf_modelencoderlayers0residual_connections0normbias);
     auto hostBuf_modelprojection_layerlinear_chargebias = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear_chargebias, 1);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear_chargebias, hostBuf_modelprojection_layerlinear_chargebias);
     auto hostBuf_val_5 = alpaka::createView(hostAcc, tensor_val_5, 1280);
     alpaka::memcpy(queue, deviceBuf_val_5, hostBuf_val_5);
     auto hostBuf_modelprojection_layerlinear_regressor_ptbias = alpaka::createView(hostAcc, tensor_modelprojection_layerlinear_regressor_ptbias, 1);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerlinear_regressor_ptbias, hostBuf_modelprojection_layerlinear_regressor_ptbias);
     auto hostBuf_val_13 = alpaka::createView(hostAcc, tensor_val_13, 65536);
     alpaka::memcpy(queue, deviceBuf_val_13, hostBuf_val_13);
     auto hostBuf_val_15 = alpaka::createView(hostAcc, tensor_val_15, 49152);
     alpaka::memcpy(queue, deviceBuf_val_15, hostBuf_val_15);
     auto hostBuf_val_55 = alpaka::createView(hostAcc, tensor_val_55, 1);
     alpaka::memcpy(queue, deviceBuf_val_55, hostBuf_val_55);
     auto hostBuf_val_77 = alpaka::createView(hostAcc, tensor_val_77, 64);
     alpaka::memcpy(queue, deviceBuf_val_77, hostBuf_val_77);
     auto hostBuf_mul_55 = alpaka::createView(hostAcc, tensor_mul_55, 1);
     alpaka::memcpy(queue, deviceBuf_mul_55, hostBuf_mul_55);
     auto hostBuf_val_80 = alpaka::createView(hostAcc, tensor_val_80, 4);
     alpaka::memcpy(queue, deviceBuf_val_80, hostBuf_val_80);
     auto hostBuf_modelencoderlayers0self_attention_blockout_projweight = alpaka::createView(hostAcc, tensor_modelencoderlayers0self_attention_blockout_projweight, 16384);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0self_attention_blockout_projweight, hostBuf_modelencoderlayers0self_attention_blockout_projweight);
     auto hostBuf_modelparticle_embedembed7bias = alpaka::createView(hostAcc, tensor_modelparticle_embedembed7bias, 128);
     alpaka::memcpy(queue, deviceBuf_modelparticle_embedembed7bias, hostBuf_modelparticle_embedembed7bias);
     auto hostBuf_val_23 = alpaka::createView(hostAcc, tensor_val_23, 1);
     alpaka::memcpy(queue, deviceBuf_val_23, hostBuf_val_23);
     auto hostBuf_val_24 = alpaka::createView(hostAcc, tensor_val_24, 1);
     alpaka::memcpy(queue, deviceBuf_val_24, hostBuf_val_24);
     auto hostBuf_val_26 = alpaka::createView(hostAcc, tensor_val_26, 1);
     alpaka::memcpy(queue, deviceBuf_val_26, hostBuf_val_26);
     auto hostBuf_modelprojection_layerpooling_layerscore_layer0bias = alpaka::createView(hostAcc, tensor_modelprojection_layerpooling_layerscore_layer0bias, 64);
     alpaka::memcpy(queue, deviceBuf_modelprojection_layerpooling_layerscore_layer0bias, hostBuf_modelprojection_layerpooling_layerscore_layer0bias);
     auto hostBuf_val_59 = alpaka::createView(hostAcc, tensor_val_59, 1);
     alpaka::memcpy(queue, deviceBuf_val_59, hostBuf_val_59);
     auto hostBuf_val_60 = alpaka::createView(hostAcc, tensor_val_60, 1);
     alpaka::memcpy(queue, deviceBuf_val_60, hostBuf_val_60);
     auto hostBuf_modelencoderlayers0feed_forward_blocklinear_2bias = alpaka::createView(hostAcc, tensor_modelencoderlayers0feed_forward_blocklinear_2bias, 128);
     alpaka::memcpy(queue, deviceBuf_modelencoderlayers0feed_forward_blocklinear_2bias, hostBuf_modelencoderlayers0feed_forward_blocklinear_2bias);
     auto hostBuf_val_68 = alpaka::createView(hostAcc, tensor_val_68, 65536);
     alpaka::memcpy(queue, deviceBuf_val_68, hostBuf_val_68);
     auto hostBuf_val_79 = alpaka::createView(hostAcc, tensor_val_79, 1);
     alpaka::memcpy(queue, deviceBuf_val_79, hostBuf_val_79);
//---- allocate unified intermediate tensor pool

   // --- Unified tensor pool: lifetime-aware packing ---
   {
      // +8192 bytes appended after the packed region: extra slot for the three
      // simultaneously-live final outputs (see charge_logit assignment below)
      constexpr std::size_t kConstPoolBytes = 169082880 + 8192;
      std::size_t poolTotalBytes = kConstPoolBytes;

      // Allocate unified pool buffer
      if (poolTotalBytes > 0)
         fUnifiedPool = alpaka::allocBuf<uint8_t, Idx>(devAcc, Ext1D::all(Idx{poolTotalBytes}));
      auto* poolBase = (poolTotalBytes > 0) ? alpaka::getPtrNative(fUnifiedPool) : nullptr;
      fUnifiedPoolSize = poolTotalBytes;

      // Assign tensor views into unified pool
      deviceBuf_reg_pt = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{2048})};
      deviceBuf_class_logit = ViewF1D{reinterpret_cast<float*>(poolBase + 24576), devAcc, Ext1D::all(Idx{2048})};
      deviceBuf_linear_12 = ViewF1D{reinterpret_cast<float*>(poolBase + 8192), devAcc, Ext1D::all(Idx{2048})};
      deviceBuf_sum_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 17039360), devAcc, Ext1D::all(Idx{262144})};
      deviceBuf_linear_8 = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{32768})};
      deviceBuf_linear_7 = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{2097152})};
      deviceBuf_softmax_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 16908288), devAcc, Ext1D::all(Idx{32768})};
      deviceBuf_sub_100 = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_transpose_4 = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_relu_4 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{131072})};
      deviceBuf_select = ViewF1D{reinterpret_cast<float*>(poolBase + 134217728), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_relu_5 = ViewF1D{reinterpret_cast<float*>(poolBase + 524288), devAcc, Ext1D::all(Idx{65536})};
      deviceBuf_select_2 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_transpose_5 = ViewF1D{reinterpret_cast<float*>(poolBase + 33554432), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_select_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 150994944), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_relu = ViewF1D{reinterpret_cast<float*>(poolBase + 1310720), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_linear_13 = ViewF1D{reinterpret_cast<float*>(poolBase + 16384), devAcc, Ext1D::all(Idx{2048})};
      deviceBuf_transpose_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 83886080), devAcc, Ext1D::all(Idx{12582912})};
      deviceBuf_transpose_6 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_layer_norm = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{327680})};
      deviceBuf_softmax = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_mul_74 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      // FIX (generator defect): final outputs must not share arena slots;
      // reg_pt takes poolBase+0, so charge_logit gets the appended tail slot
      // instead of aliasing it (was: poolBase + 0)
      deviceBuf_charge_logit = ViewF1D{reinterpret_cast<float*>(poolBase + 169082880), devAcc, Ext1D::all(Idx{2048})};
      deviceBuf_transpose = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_bmm = ViewF1D{reinterpret_cast<float*>(poolBase + 50331648), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_relu_3 = ViewF1D{reinterpret_cast<float*>(poolBase + 50462720), devAcc, Ext1D::all(Idx{16777216})};
      deviceBuf_relu_2 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_linear_11 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{2048})};
      deviceBuf_linear_6 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_add_278 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_layer_norm_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 18087936), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_tanh = ViewF1D{reinterpret_cast<float*>(poolBase + 25165824), devAcc, Ext1D::all(Idx{2097152})};
      deviceBuf_relu_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 34865152), devAcc, Ext1D::all(Idx{16777216})};
      deviceBuf_div = ViewF1D{reinterpret_cast<float*>(poolBase + 16908288), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_layer_norm_2 = ViewF1D{reinterpret_cast<float*>(poolBase + 101974016), devAcc, Ext1D::all(Idx{16777216})};
      deviceBuf_linear_4 = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_mean = ViewF1D{reinterpret_cast<float*>(poolBase + 16777216), devAcc, Ext1D::all(Idx{32768})};
      deviceBuf_transpose_7 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_linear_3 = ViewF1D{reinterpret_cast<float*>(poolBase + 33554432), devAcc, Ext1D::all(Idx{12582912})};
      deviceBuf_bmm_1 = ViewF1D{reinterpret_cast<float*>(poolBase + 33554432), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_add_189 = ViewF1D{reinterpret_cast<float*>(poolBase + 33685504), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_sub_81 = ViewF1D{reinterpret_cast<float*>(poolBase + 50462720), devAcc, Ext1D::all(Idx{4194304})};
      deviceBuf_add_224 = ViewF1D{reinterpret_cast<float*>(poolBase + 0), devAcc, Ext1D::all(Idx{4194304})};
   }

if constexpr (!std::is_same_v<tagAcc, alpaka::TagCpuSerial>) {
   blas.addOperationConfig(128, 32768, 10, 128, 10, 128, 'n', 'n', Epilogue::ReluBias);
   
   blas.addOperationConfig(512, 32768, 128, 512, 128, 512, 'n', 'n', Epilogue::ReluBias);
   
   blas.addOperationConfig(128, 32768, 512, 128, 512, 128, 'n', 'n', Epilogue::ReluBias);
   
   blas.addOperationConfig(384, 32768, 128, 384, 128, 384, 'n', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(128, 32768, 128, 128, 128, 128, 't', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(512, 32768, 128, 512, 128, 512, 'n', 'n', Epilogue::ReluBias);
   
   blas.addOperationConfig(128, 32768, 512, 128, 512, 128, 'n', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(64, 32768, 128, 64, 128, 64, 'n', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(1, 32768, 64, 1, 64, 1, 'n', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(64, 2048, 128, 128, 128, 64, 't', 'n', Epilogue::ReluBias);
   
   blas.addOperationConfig(32, 2048, 64, 64, 64, 32, 't', 'n', Epilogue::ReluBias);
   
   blas.addOperationConfig(1, 2048, 32, 32, 32, 1, 't', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(1, 2048, 32, 32, 32, 1, 't', 'n', Epilogue::Bias);
   
   blas.addOperationConfig(1, 2048, 32, 32, 32, 1, 't', 'n', Epilogue::Bias);
}

alpaka::wait(queue);
}

public:


Session(std::string filename = "STT_ort_relu_fp32_b2048_s16.dat") {
   InitSession(filename);
}

Session(QueueAcc& extQueue, std::string filename = "STT_ort_relu_fp32_b2048_s16.dat")
    : queue(extQueue), blas(queue)
{
   InitSession(filename);
}

   LayerNormKernel_op_3 layerNormKernel_op_3;
GemmBiasAddKernel gemmBiasAddKernel;
   ReluKernel reluKernel;
   LayerNormKernel_op_6 layerNormKernel_op_6;
   LayerNormKernel_op_9 layerNormKernel_op_9;
   TransposeKernel_op_12 transposeKernel_12;
   FusedEltwiseKernel_14_15_16 fusedEltwiseKernel_14_15_16;
   GatherKernel_op_18 gatherKernel_op_18;
   GatherKernel_op_19 gatherKernel_op_19;
   GatherKernel_op_20 gatherKernel_op_20;
   FusedEltwiseKernel_24_25_29 fusedEltwiseKernel_24_25_29;
   FusedEltwiseKernel_26_30 fusedEltwiseKernel_26_30;
   FusedEltwiseKernel_27_28 fusedEltwiseKernel_27_28;
   SoftmaxKernel_op_32 softmaxKernel_op_32;
   TransposeKernel_op_34 transposeKernel_34;
   FusedEltwiseKernel_38_39 fusedEltwiseKernel_38_39;
   ReduceKernel_ReduceMean_mean reduceKernel_ReduceMean_mean;
   FusedEltwiseKernel_41_42_43_44_45_46_47_48 fusedEltwiseKernel_41_42_43_44_45_46_47_48;
   FusedEltwiseKernel_49_50_51_52 fusedEltwiseKernel_49_50_51_52;
   FusedEltwiseKernel_53_54_55_56_57_58_59_60 fusedEltwiseKernel_53_54_55_56_57_58_59_60;
   FusedEltwiseKernel_64_65_66 fusedEltwiseKernel_64_65_66;
   FusedEltwiseKernel_67_68_69_70_71_72_73_74 fusedEltwiseKernel_67_68_69_70_71_72_73_74;
   TanhKernel tanhKernel;
   SoftmaxKernel_op_78 softmaxKernel_op_78;
   FusedEltwiseKernel_79_80 fusedEltwiseKernel_79_80;
   FusedEltwiseKernel_84_89 fusedEltwiseKernel_84_89;
   FusedEltwiseKernel_86_91 fusedEltwiseKernel_86_91;
   FusedEltwiseKernel_88_90 fusedEltwiseKernel_88_90;

void _infer_impl(ViewConstF1D const& deviceBuf_particles,ViewConstF1D const& deviceBuf_padding_masks);

void infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs);

std::tuple<ViewF1D,ViewF1D,ViewF1D> infer(BufF1D const deviceBuf_particles,BufF1D const deviceBuf_padding_masks);

std::tuple<ViewF1D,ViewF1D,ViewF1D> infer_nosync(BufF1D const deviceBuf_particles,BufF1D const deviceBuf_padding_masks);

void resetState(QueueAcc& queue) {
   alpaka::memset(queue, deviceBuf_reg_pt, 0);
   alpaka::memset(queue, deviceBuf_class_logit, 0);
   alpaka::memset(queue, deviceBuf_linear_12, 0);
   alpaka::memset(queue, deviceBuf_sum_1, 0);
   alpaka::memset(queue, deviceBuf_linear_8, 0);
   alpaka::memset(queue, deviceBuf_linear_7, 0);
   alpaka::memset(queue, deviceBuf_softmax_1, 0);
   alpaka::memset(queue, deviceBuf_sub_100, 0);
   alpaka::memset(queue, deviceBuf_transpose_4, 0);
   alpaka::memset(queue, deviceBuf_relu_4, 0);
   alpaka::memset(queue, deviceBuf_select, 0);
   alpaka::memset(queue, deviceBuf_relu_5, 0);
   alpaka::memset(queue, deviceBuf_select_2, 0);
   alpaka::memset(queue, deviceBuf_transpose_5, 0);
   alpaka::memset(queue, deviceBuf_select_1, 0);
   alpaka::memset(queue, deviceBuf_relu, 0);
   alpaka::memset(queue, deviceBuf_linear_13, 0);
   alpaka::memset(queue, deviceBuf_transpose_1, 0);
   alpaka::memset(queue, deviceBuf_transpose_6, 0);
   alpaka::memset(queue, deviceBuf_layer_norm, 0);
   alpaka::memset(queue, deviceBuf_linear_1, 0);
   alpaka::memset(queue, deviceBuf_softmax, 0);
   alpaka::memset(queue, deviceBuf_mul_74, 0);
   alpaka::memset(queue, deviceBuf_linear, 0);
   alpaka::memset(queue, deviceBuf_linear_5, 0);
   alpaka::memset(queue, deviceBuf_charge_logit, 0);
   alpaka::memset(queue, deviceBuf_transpose, 0);
   alpaka::memset(queue, deviceBuf_bmm, 0);
   alpaka::memset(queue, deviceBuf_linear_2, 0);
   alpaka::memset(queue, deviceBuf_relu_3, 0);
   alpaka::memset(queue, deviceBuf_relu_2, 0);
   alpaka::memset(queue, deviceBuf_linear_11, 0);
   alpaka::memset(queue, deviceBuf_linear_6, 0);
   alpaka::memset(queue, deviceBuf_add_278, 0);
   alpaka::memset(queue, deviceBuf_layer_norm_1, 0);
   alpaka::memset(queue, deviceBuf_tanh, 0);
   alpaka::memset(queue, deviceBuf_relu_1, 0);
   alpaka::memset(queue, deviceBuf_div, 0);
   alpaka::memset(queue, deviceBuf_layer_norm_2, 0);
   alpaka::memset(queue, deviceBuf_linear_4, 0);
   alpaka::memset(queue, deviceBuf_mean, 0);
   alpaka::memset(queue, deviceBuf_transpose_7, 0);
   alpaka::memset(queue, deviceBuf_linear_3, 0);
   alpaka::memset(queue, deviceBuf_bmm_1, 0);
   alpaka::memset(queue, deviceBuf_add_189, 0);
   alpaka::memset(queue, deviceBuf_sub_81, 0);
   alpaka::memset(queue, deviceBuf_add_224, 0);
   alpaka::wait(queue);
}
};   // end of Session

// Fully instantiate templated Session struct only inside the .dev.cc file.
// This forces kernel calls not to come from the .cc producer file.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
extern template struct Session<alpaka::TagGpuCudaRt>;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
extern template struct Session<alpaka::TagGpuHipRt>;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
extern template struct Session<alpaka::TagCpuSerial>;
#endif

} // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::SOFIE_STT_ort_relu_fp32_b2048_s16

#endif