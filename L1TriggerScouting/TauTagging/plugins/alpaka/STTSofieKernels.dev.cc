#include "L1TriggerScouting/TauTagging/plugins/alpaka/STTSofieKernels.h"

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

} // namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::SOFIE_STT_ort_relu_fp32_b2048_s16

#endif  // SOFIE_STT_ORT_RELU_FP32_B2048_S16_HXX
