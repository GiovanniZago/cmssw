#include "L1TriggerScouting/TauTagging/plugins/alpaka/STTSofieKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::l1sc::SOFIE_STT_ort_relu_fp32_b2048_s16 {

template <typename Acc>
void Session<Acc>::_infer_impl(ViewConstF1D const& deviceBuf_particles,ViewConstF1D const& deviceBuf_padding_masks){

//------ LAYERNORM_GPU_ALPAKA
   alpaka::WorkDivMembers<Dim, Idx> workDiv_op_3(
      Vec::all(Idx{32768}),
      Vec::all(Idx{16u}),
      Vec::all(Idx{1u}));
   alpaka::exec<Acc>(queue, workDiv_op_3, layerNormKernel_op_3, alpaka::getPtrNative(deviceBuf_particles), alpaka::getPtrNative(deviceBuf_modelparticle_embedembed0weight), alpaka::getPtrNative(deviceBuf_modelparticle_embedembed0bias), alpaka::getPtrNative(deviceBuf_layer_norm), static_cast<Idx>(32768));

//--------- Gemm_GPU_ALPAKA
   char op_4_transA = 'n';
   char op_4_transB = 'n';
   int op_4_m = 16;
   int op_4_n = 128;
   int op_4_k = 10;
   float op_4_alpha = 1;
   float op_4_beta = 0;
   blas.gemmrelu(op_4_transB, op_4_transA, op_4_n, 32768, op_4_k, op_4_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_5)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_layer_norm)), op_4_beta, alpaka::getPtrNative(deviceBuf_modelparticle_embedembed1bias), alpaka::getPtrNative(deviceBuf_relu));

//------ LAYERNORM_GPU_ALPAKA
   alpaka::WorkDivMembers<Dim, Idx> workDiv_op_6(
      Vec::all(Idx{32768}),
      Vec::all(Idx{128u}),
      Vec::all(Idx{1u}));
   alpaka::exec<Acc>(queue, workDiv_op_6, layerNormKernel_op_6, alpaka::getPtrNative(deviceBuf_relu), alpaka::getPtrNative(deviceBuf_modelparticle_embedembed3weight), alpaka::getPtrNative(deviceBuf_modelparticle_embedembed3bias), alpaka::getPtrNative(deviceBuf_layer_norm_1), static_cast<Idx>(32768));

//--------- Gemm_GPU_ALPAKA
   char op_7_transA = 'n';
   char op_7_transB = 'n';
   int op_7_m = 16;
   int op_7_n = 512;
   int op_7_k = 128;
   float op_7_alpha = 1;
   float op_7_beta = 0;
   blas.gemmrelu(op_7_transB, op_7_transA, op_7_n, 32768, op_7_k, op_7_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_9)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_layer_norm_1)), op_7_beta, alpaka::getPtrNative(deviceBuf_modelparticle_embedembed4bias), alpaka::getPtrNative(deviceBuf_relu_1));

//------ LAYERNORM_GPU_ALPAKA
   alpaka::WorkDivMembers<Dim, Idx> workDiv_op_9(
      Vec::all(Idx{32768}),
      Vec::all(Idx{256u}),
      Vec::all(Idx{1u}));
   alpaka::exec<Acc>(queue, workDiv_op_9, layerNormKernel_op_9, alpaka::getPtrNative(deviceBuf_relu_1), alpaka::getPtrNative(deviceBuf_modelparticle_embedembed6weight), alpaka::getPtrNative(deviceBuf_modelparticle_embedembed6bias), alpaka::getPtrNative(deviceBuf_layer_norm_2), static_cast<Idx>(32768));

//--------- Gemm_GPU_ALPAKA
   char op_10_transA = 'n';
   char op_10_transB = 'n';
   int op_10_m = 16;
   int op_10_n = 128;
   int op_10_k = 512;
   float op_10_alpha = 1;
   float op_10_beta = 0;
   blas.gemmrelu(op_10_transB, op_10_transA, op_10_n, 32768, op_10_k, op_10_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_13)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_layer_norm_2)), op_10_beta, alpaka::getPtrNative(deviceBuf_modelparticle_embedembed7bias), alpaka::getPtrNative(deviceBuf_relu_2));

//------ TRANSPOSE_GPU_ALPAKA
   auto const elementsPerThread_transpose = Vec::all(static_cast<Idx>(1));
   auto const elementsPerGrid_transpose = Vec::all(Idx{4194304});
   auto const workDiv_transpose = sofie_workdiv(elementsPerGrid_transpose);
   auto task_12 = alpaka::createTaskKernel<Acc>(workDiv_transpose, transposeKernel_12, alpaka::getPtrNative(deviceBuf_relu_2), alpaka::getPtrNative(deviceBuf_transpose), static_cast<Idx>(4194304));
   alpaka::enqueue(queue, task_12);

//--------- Gemm_GPU_ALPAKA
   char op_13_transA = 'n';
   char op_13_transB = 'n';
   int op_13_m = 2048;
   int op_13_n = 384;
   int op_13_k = 128;
   float op_13_alpha = 1;
   float op_13_beta = 0;
   blas.gemm(op_13_transB, op_13_transA, op_13_n, 32768, op_13_k, op_13_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_15)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_transpose)), op_13_beta, alpaka::getPtrNative(deviceBuf_modelencoderlayers0self_attention_blockin_proj_bias), alpaka::getPtrNative(deviceBuf_linear_3));

//------ FUSED_ELTWISE_GPU_ALPAKA_14_15_16
   {
      auto const elementsPerThread_fused_14_15_16 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_14_15_16 = Vec::all(Idx{12582912});
      auto const workDiv_fused_14_15_16 = sofie_workdiv(elementsPerGrid_fused_14_15_16);
      auto task_fused_14_15_16 = alpaka::createTaskKernel<Acc>(workDiv_fused_14_15_16, fusedEltwiseKernel_14_15_16, alpaka::getPtrNative(deviceBuf_linear_3), alpaka::getPtrNative(deviceBuf_transpose_1), static_cast<Idx>(12582912));
      alpaka::enqueue(queue, task_fused_14_15_16);
   }
   ///------- Squeeze operator op_17
   auto deviceBuf_squeeze = alpaka::createView(devAcc, alpaka::getPtrNative(deviceBuf_transpose_1), static_cast<Idx>(12582912));

//------ GATHER_GPU_ALPAKA
   auto const elementsPerThread_op_18 = Vec::all(static_cast<Idx>(1));
   auto const elementsPerGrid_op_18 = Vec::all(Idx{4194304});
   auto const workDiv_op_18 = sofie_workdiv(elementsPerGrid_op_18);
   auto task_op_18 = alpaka::createTaskKernel<Acc>(workDiv_op_18, gatherKernel_op_18, alpaka::getPtrNative(deviceBuf_squeeze), alpaka::getPtrNative(deviceBuf_val_25), alpaka::getPtrNative(deviceBuf_select), static_cast<Idx>(4194304));
   alpaka::enqueue(queue, task_op_18);

//------ GATHER_GPU_ALPAKA
   auto const elementsPerThread_op_19 = Vec::all(static_cast<Idx>(1));
   auto const elementsPerGrid_op_19 = Vec::all(Idx{4194304});
   auto const workDiv_op_19 = sofie_workdiv(elementsPerGrid_op_19);
   auto task_op_19 = alpaka::createTaskKernel<Acc>(workDiv_op_19, gatherKernel_op_19, alpaka::getPtrNative(deviceBuf_squeeze), alpaka::getPtrNative(deviceBuf_val_2), alpaka::getPtrNative(deviceBuf_select_1), static_cast<Idx>(4194304));
   alpaka::enqueue(queue, task_op_19);

//------ GATHER_GPU_ALPAKA
   auto const elementsPerThread_op_20 = Vec::all(static_cast<Idx>(1));
   auto const elementsPerGrid_op_20 = Vec::all(Idx{4194304});
   auto const workDiv_op_20 = sofie_workdiv(elementsPerGrid_op_20);
   auto task_op_20 = alpaka::createTaskKernel<Acc>(workDiv_op_20, gatherKernel_op_20, alpaka::getPtrNative(deviceBuf_squeeze), alpaka::getPtrNative(deviceBuf_val_26), alpaka::getPtrNative(deviceBuf_select_2), static_cast<Idx>(4194304));
   alpaka::enqueue(queue, task_op_20);

//------ FUSED_ELTWISE_GPU_ALPAKA_27_28
   {
      auto const elementsPerThread_fused_27_28 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_27_28 = Vec::all(Idx{4194304});
      auto const workDiv_fused_27_28 = sofie_workdiv(elementsPerGrid_fused_27_28);
      auto task_fused_27_28 = alpaka::createTaskKernel<Acc>(workDiv_fused_27_28, fusedEltwiseKernel_27_28, alpaka::getPtrNative(deviceBuf_select_2), alpaka::getPtrNative(deviceBuf_transpose_4), static_cast<Idx>(4194304));
      alpaka::enqueue(queue, task_fused_27_28);
   }

//------ FUSED_ELTWISE_GPU_ALPAKA_24_25_29
   {
      auto const elementsPerThread_fused_24_25_29 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_24_25_29 = Vec::all(Idx{4194304});
      auto const workDiv_fused_24_25_29 = sofie_workdiv(elementsPerGrid_fused_24_25_29);
      auto task_fused_24_25_29 = alpaka::createTaskKernel<Acc>(workDiv_fused_24_25_29, fusedEltwiseKernel_24_25_29, alpaka::getPtrNative(deviceBuf_select), alpaka::getPtrNative(deviceBuf_val_43), alpaka::getPtrNative(deviceBuf_mul_74), static_cast<Idx>(4194304));
      alpaka::enqueue(queue, task_fused_24_25_29);
   }

//------ FUSED_ELTWISE_GPU_ALPAKA_26_30
   {
      auto const elementsPerThread_fused_26_30 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_26_30 = Vec::all(Idx{4194304});
      auto const workDiv_fused_26_30 = sofie_workdiv(elementsPerGrid_fused_26_30);
      auto task_fused_26_30 = alpaka::createTaskKernel<Acc>(workDiv_fused_26_30, fusedEltwiseKernel_26_30, alpaka::getPtrNative(deviceBuf_select_1), alpaka::getPtrNative(deviceBuf_transpose_5), static_cast<Idx>(4194304));
      alpaka::enqueue(queue, task_fused_26_30);
   }

//--------- Gemm_GPU_ALPAKA
   char op_31_transA = 'n';
   char op_31_transB = 'n';
   int op_31_m = 16;
   int op_31_n = 16;
   int op_31_k = 16;
   float op_31_alpha = 1;
   float op_31_beta = 0;
   blas.gemmStridedBatched(op_31_transB, op_31_transA, 16, 16, 16, op_31_alpha, alpaka::getPtrNative(deviceBuf_transpose_5), 16, 256, alpaka::getPtrNative(deviceBuf_mul_74), 16, 256, op_31_beta, alpaka::getPtrNative(deviceBuf_bmm), 16, 256, 16384);

//------ SOFTMAX_GPU_ALPAKA
   alpaka::WorkDivMembers<Dim, Idx> workDiv_op_32(
      Vec::all(static_cast<Idx>(262144)),
      Vec::all(Idx{32u}),
      Vec::all(Idx{1u}));
   auto task_op_32 = alpaka::createTaskKernel<Acc>(workDiv_op_32, softmaxKernel_op_32, alpaka::getPtrNative(deviceBuf_bmm), alpaka::getPtrNative(deviceBuf_softmax), static_cast<Idx>(262144));
   alpaka::enqueue(queue, task_op_32);

//--------- Gemm_GPU_ALPAKA
   char op_33_transA = 'n';
   char op_33_transB = 'n';
   int op_33_m = 16;
   int op_33_n = 16;
   int op_33_k = 16;
   float op_33_alpha = 1;
   float op_33_beta = 0;
   blas.gemmStridedBatched(op_33_transB, op_33_transA, 16, 16, 16, op_33_alpha, alpaka::getPtrNative(deviceBuf_transpose_4), 16, 256, alpaka::getPtrNative(deviceBuf_softmax), 16, 256, op_33_beta, alpaka::getPtrNative(deviceBuf_bmm_1), 16, 256, 16384);

//------ TRANSPOSE_GPU_ALPAKA
   auto const elementsPerThread_transpose_6 = Vec::all(static_cast<Idx>(1));
   auto const elementsPerGrid_transpose_6 = Vec::all(Idx{4194304});
   auto const workDiv_transpose_6 = sofie_workdiv(elementsPerGrid_transpose_6);
   auto task_34 = alpaka::createTaskKernel<Acc>(workDiv_transpose_6, transposeKernel_34, alpaka::getPtrNative(deviceBuf_bmm_1), alpaka::getPtrNative(deviceBuf_transpose_6), static_cast<Idx>(4194304));
   alpaka::enqueue(queue, task_34);
   ///------- Reshape operator op_35
   auto deviceBuf_view_4 = alpaka::createView(devAcc, alpaka::getPtrNative(deviceBuf_transpose_6), static_cast<Idx>(4194304));

//--------- Gemm_GPU_ALPAKA
   char op_36_transA = 'n';
   char op_36_transB = 't';
   int op_36_m = 32768;
   int op_36_n = 128;
   int op_36_k = 128;
   float op_36_alpha = 1;
   float op_36_beta = 0;
   blas.gemm(op_36_transB, op_36_transA, op_36_n, op_36_m, op_36_k, op_36_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_modelencoderlayers0self_attention_blockout_projweight)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_view_4)), op_36_beta, alpaka::getPtrNative(deviceBuf_modelencoderlayers0self_attention_blockout_projbias), alpaka::getPtrNative(deviceBuf_linear_4));

//------ FUSED_ELTWISE_GPU_ALPAKA_38_39
   {
      auto const elementsPerThread_fused_38_39 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_38_39 = Vec::all(Idx{4194304});
      auto const workDiv_fused_38_39 = sofie_workdiv(elementsPerGrid_fused_38_39);
      auto task_fused_38_39 = alpaka::createTaskKernel<Acc>(workDiv_fused_38_39, fusedEltwiseKernel_38_39, alpaka::getPtrNative(deviceBuf_linear_4), alpaka::getPtrNative(deviceBuf_transpose_7), static_cast<Idx>(4194304));
      alpaka::enqueue(queue, task_fused_38_39);
   }

//------ ReduceMean_GPU_ALPAKA
   alpaka::WorkDivMembers<Dim, Idx> workDiv_mean(
      Vec::all(Idx{32768}),
      Vec::all(Idx{128u}),
      Vec::all(Idx{1u}));
   alpaka::exec<Acc>(queue, workDiv_mean, reduceKernel_ReduceMean_mean, alpaka::getPtrNative(deviceBuf_transpose_7), alpaka::getPtrNative(deviceBuf_mean), static_cast<std::size_t>((4194304) / (32768)), static_cast<std::size_t>(32768));

//------ FUSED_REDUCTION_GPU_ALPAKA_41_42_43_44_45_46_47_48
   {
      alpaka::WorkDivMembers<Dim, Idx> workDiv_fused_41_42_43_44_45_46_47_48(
         Vec::all(Idx{32768u}),
         Vec::all(Idx{128u}),
         Vec::all(Idx{1u}));
      auto task_fused_41_42_43_44_45_46_47_48 = alpaka::createTaskKernel<Acc>(workDiv_fused_41_42_43_44_45_46_47_48, fusedEltwiseKernel_41_42_43_44_45_46_47_48, alpaka::getPtrNative(deviceBuf_transpose_7), alpaka::getPtrNative(deviceBuf_mean), alpaka::getPtrNative(deviceBuf_val_56), alpaka::getPtrNative(deviceBuf_val_59), alpaka::getPtrNative(deviceBuf_modelencoderlayers0residual_connections0normalpha), alpaka::getPtrNative(deviceBuf_val_60), alpaka::getPtrNative(deviceBuf_div));
      alpaka::enqueue(queue, task_fused_41_42_43_44_45_46_47_48);
   }

//------ FUSED_REDUCTION_GPU_ALPAKA_49_50_51_52
   {
      alpaka::WorkDivMembers<Dim, Idx> workDiv_fused_49_50_51_52(
         Vec::all(Idx{32768u}),
         Vec::all(Idx{128u}),
         Vec::all(Idx{1u}));
      auto task_fused_49_50_51_52 = alpaka::createTaskKernel<Acc>(workDiv_fused_49_50_51_52, fusedEltwiseKernel_49_50_51_52, alpaka::getPtrNative(deviceBuf_div), alpaka::getPtrNative(deviceBuf_modelencoderlayers0residual_connections0normbias), alpaka::getPtrNative(deviceBuf_transpose_7), alpaka::getPtrNative(deviceBuf_add_189), alpaka::getPtrNative(deviceBuf_sub_81));
      alpaka::enqueue(queue, task_fused_49_50_51_52);
   }

//------ FUSED_REDUCTION_GPU_ALPAKA_53_54_55_56_57_58_59_60
   {
      alpaka::WorkDivMembers<Dim, Idx> workDiv_fused_53_54_55_56_57_58_59_60(
         Vec::all(Idx{32768u}),
         Vec::all(Idx{128u}),
         Vec::all(Idx{1u}));
      auto task_fused_53_54_55_56_57_58_59_60 = alpaka::createTaskKernel<Acc>(workDiv_fused_53_54_55_56_57_58_59_60, fusedEltwiseKernel_53_54_55_56_57_58_59_60, alpaka::getPtrNative(deviceBuf_sub_81), alpaka::getPtrNative(deviceBuf_val_56), alpaka::getPtrNative(deviceBuf_val_59), alpaka::getPtrNative(deviceBuf_modelencoderlayers0residual_connections1normalpha), alpaka::getPtrNative(deviceBuf_val_60), alpaka::getPtrNative(deviceBuf_modelencoderlayers0residual_connections1normbias), alpaka::getPtrNative(deviceBuf_add_224));
      alpaka::enqueue(queue, task_fused_53_54_55_56_57_58_59_60);
   }

//--------- Gemm_GPU_ALPAKA
   char op_61_transA = 'n';
   char op_61_transB = 'n';
   int op_61_m = 16;
   int op_61_n = 512;
   int op_61_k = 128;
   float op_61_alpha = 1;
   float op_61_beta = 0;
   blas.gemmrelu(op_61_transB, op_61_transA, op_61_n, 32768, op_61_k, op_61_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_66)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_add_224)), op_61_beta, alpaka::getPtrNative(deviceBuf_modelencoderlayers0feed_forward_blocklinear_1bias), alpaka::getPtrNative(deviceBuf_relu_3));

//--------- Gemm_GPU_ALPAKA
   char op_63_transA = 'n';
   char op_63_transB = 'n';
   int op_63_m = 16;
   int op_63_n = 128;
   int op_63_k = 512;
   float op_63_alpha = 1;
   float op_63_beta = 0;
   blas.gemm(op_63_transB, op_63_transA, op_63_n, 32768, op_63_k, op_63_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_68)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_relu_3)), op_63_beta, alpaka::getPtrNative(deviceBuf_modelencoderlayers0feed_forward_blocklinear_2bias), alpaka::getPtrNative(deviceBuf_linear_6));

//------ FUSED_REDUCTION_GPU_ALPAKA_64_65_66
   {
      alpaka::WorkDivMembers<Dim, Idx> workDiv_fused_64_65_66(
         Vec::all(Idx{32768u}),
         Vec::all(Idx{128u}),
         Vec::all(Idx{1u}));
      auto task_fused_64_65_66 = alpaka::createTaskKernel<Acc>(workDiv_fused_64_65_66, fusedEltwiseKernel_64_65_66, alpaka::getPtrNative(deviceBuf_add_189), alpaka::getPtrNative(deviceBuf_linear_6), alpaka::getPtrNative(deviceBuf_sub_100));
      alpaka::enqueue(queue, task_fused_64_65_66);
   }

//------ FUSED_REDUCTION_GPU_ALPAKA_67_68_69_70_71_72_73_74
   {
      alpaka::WorkDivMembers<Dim, Idx> workDiv_fused_67_68_69_70_71_72_73_74(
         Vec::all(Idx{32768u}),
         Vec::all(Idx{128u}),
         Vec::all(Idx{1u}));
      auto task_fused_67_68_69_70_71_72_73_74 = alpaka::createTaskKernel<Acc>(workDiv_fused_67_68_69_70_71_72_73_74, fusedEltwiseKernel_67_68_69_70_71_72_73_74, alpaka::getPtrNative(deviceBuf_sub_100), alpaka::getPtrNative(deviceBuf_val_56), alpaka::getPtrNative(deviceBuf_val_59), alpaka::getPtrNative(deviceBuf_modelencodernormalpha), alpaka::getPtrNative(deviceBuf_val_60), alpaka::getPtrNative(deviceBuf_modelencodernormbias), alpaka::getPtrNative(deviceBuf_add_278));
      alpaka::enqueue(queue, task_fused_67_68_69_70_71_72_73_74);
   }

//--------- Gemm_GPU_ALPAKA
   char op_75_transA = 'n';
   char op_75_transB = 'n';
   int op_75_m = 16;
   int op_75_n = 64;
   int op_75_k = 128;
   float op_75_alpha = 1;
   float op_75_beta = 0;
   blas.gemm(op_75_transB, op_75_transA, op_75_n, 32768, op_75_k, op_75_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_75)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_add_278)), op_75_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerpooling_layerscore_layer0bias), alpaka::getPtrNative(deviceBuf_linear_7));

//------ TANH_GPU_ALPAKA
   auto const elementsPerThread_linear_7 = Vec::all(static_cast<Idx>(1));
   auto const elementsPerGrid_linear_7 = Vec::all(Idx{2097152});
   auto const workDiv_linear_7 = sofie_workdiv(elementsPerGrid_linear_7);
   auto task_op_76 = alpaka::createTaskKernel<Acc>(workDiv_linear_7, tanhKernel, alpaka::getPtrNative(deviceBuf_linear_7), alpaka::getPtrNative(deviceBuf_tanh), static_cast<Idx>(2097152));
   alpaka::enqueue(queue, task_op_76);

//--------- Gemm_GPU_ALPAKA
   char op_77_transA = 'n';
   char op_77_transB = 'n';
   int op_77_m = 16;
   int op_77_n = 1;
   int op_77_k = 64;
   float op_77_alpha = 1;
   float op_77_beta = 0;
   blas.gemm(op_77_transB, op_77_transA, op_77_n, 32768, op_77_k, op_77_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_val_77)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_tanh)), op_77_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerpooling_layerscore_layer2bias), alpaka::getPtrNative(deviceBuf_linear_8));

//------ SOFTMAX_GPU_ALPAKA
   alpaka::WorkDivMembers<Dim, Idx> workDiv_op_78(
      Vec::all(static_cast<Idx>(2048)),
      Vec::all(Idx{32u}),
      Vec::all(Idx{1u}));
   auto task_op_78 = alpaka::createTaskKernel<Acc>(workDiv_op_78, softmaxKernel_op_78, alpaka::getPtrNative(deviceBuf_linear_8), alpaka::getPtrNative(deviceBuf_softmax_1), static_cast<Idx>(2048));
   alpaka::enqueue(queue, task_op_78);

//------ FUSED_REDUCTION_GPU_ALPAKA_79_80
   {
      alpaka::WorkDivMembers<Dim, Idx> workDiv_fused_79_80(
         Vec::all(Idx{262144u}),
         Vec::all(Idx{32u}),
         Vec::all(Idx{1u}));
      auto task_fused_79_80 = alpaka::createTaskKernel<Acc>(workDiv_fused_79_80, fusedEltwiseKernel_79_80, alpaka::getPtrNative(deviceBuf_softmax_1), alpaka::getPtrNative(deviceBuf_add_278), alpaka::getPtrNative(deviceBuf_sum_1));
      alpaka::enqueue(queue, task_fused_79_80);
   }

//--------- Gemm_GPU_ALPAKA
   char op_81_transA = 'n';
   char op_81_transB = 't';
   int op_81_m = 2048;
   int op_81_n = 64;
   int op_81_k = 128;
   float op_81_alpha = 1;
   float op_81_beta = 0;
   blas.gemmrelu(op_81_transB, op_81_transA, op_81_n, op_81_m, op_81_k, op_81_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear1weight)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_sum_1)), op_81_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear1bias), alpaka::getPtrNative(deviceBuf_relu_4));

//--------- Gemm_GPU_ALPAKA
   char op_82_transA = 'n';
   char op_82_transB = 't';
   int op_82_m = 2048;
   int op_82_n = 32;
   int op_82_k = 64;
   float op_82_alpha = 1;
   float op_82_beta = 0;
   blas.gemmrelu(op_82_transB, op_82_transA, op_82_n, op_82_m, op_82_k, op_82_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear2weight)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_relu_4)), op_82_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear2bias), alpaka::getPtrNative(deviceBuf_relu_5));

//--------- Gemm_GPU_ALPAKA
   char op_83_transA = 'n';
   char op_83_transB = 't';
   int op_83_m = 2048;
   int op_83_n = 1;
   int op_83_k = 32;
   float op_83_alpha = 1;
   float op_83_beta = 0;
   blas.gemm(op_83_transB, op_83_transA, op_83_n, op_83_m, op_83_k, op_83_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear_classifierweight)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_relu_5)), op_83_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear_classifierbias), alpaka::getPtrNative(deviceBuf_linear_11));

//--------- Gemm_GPU_ALPAKA
   char op_85_transA = 'n';
   char op_85_transB = 't';
   int op_85_m = 2048;
   int op_85_n = 1;
   int op_85_k = 32;
   float op_85_alpha = 1;
   float op_85_beta = 0;
   blas.gemm(op_85_transB, op_85_transA, op_85_n, op_85_m, op_85_k, op_85_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear_regressor_ptweight)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_relu_5)), op_85_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear_regressor_ptbias), alpaka::getPtrNative(deviceBuf_linear_12));

//--------- Gemm_GPU_ALPAKA
   char op_87_transA = 'n';
   char op_87_transB = 't';
   int op_87_m = 2048;
   int op_87_n = 1;
   int op_87_k = 32;
   float op_87_alpha = 1;
   float op_87_beta = 0;
   blas.gemm(op_87_transB, op_87_transA, op_87_n, op_87_m, op_87_k, op_87_alpha, static_cast<const float*>(alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear_chargeweight)), static_cast<const float*>(alpaka::getPtrNative(deviceBuf_relu_5)), op_87_beta, alpaka::getPtrNative(deviceBuf_modelprojection_layerlinear_chargebias), alpaka::getPtrNative(deviceBuf_linear_13));

//------ FUSED_ELTWISE_GPU_ALPAKA_84_89
   {
      auto const elementsPerThread_fused_84_89 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_84_89 = Vec::all(Idx{2048});
      auto const workDiv_fused_84_89 = sofie_workdiv(elementsPerGrid_fused_84_89);
      auto task_fused_84_89 = alpaka::createTaskKernel<Acc>(workDiv_fused_84_89, fusedEltwiseKernel_84_89, alpaka::getPtrNative(deviceBuf_linear_11), alpaka::getPtrNative(deviceBuf_class_logit), static_cast<Idx>(2048));
      alpaka::enqueue(queue, task_fused_84_89);
   }

//------ FUSED_ELTWISE_GPU_ALPAKA_88_90
   {
      auto const elementsPerThread_fused_88_90 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_88_90 = Vec::all(Idx{2048});
      auto const workDiv_fused_88_90 = sofie_workdiv(elementsPerGrid_fused_88_90);
      auto task_fused_88_90 = alpaka::createTaskKernel<Acc>(workDiv_fused_88_90, fusedEltwiseKernel_88_90, alpaka::getPtrNative(deviceBuf_linear_13), alpaka::getPtrNative(deviceBuf_charge_logit), static_cast<Idx>(2048));
      alpaka::enqueue(queue, task_fused_88_90);
   }

//------ FUSED_ELTWISE_GPU_ALPAKA_86_91
   {
      auto const elementsPerThread_fused_86_91 = Vec::all(static_cast<Idx>(1));
      auto const elementsPerGrid_fused_86_91 = Vec::all(Idx{2048});
      auto const workDiv_fused_86_91 = sofie_workdiv(elementsPerGrid_fused_86_91);
      auto task_fused_86_91 = alpaka::createTaskKernel<Acc>(workDiv_fused_86_91, fusedEltwiseKernel_86_91, alpaka::getPtrNative(deviceBuf_linear_12), alpaka::getPtrNative(deviceBuf_reg_pt), static_cast<Idx>(2048));
      alpaka::enqueue(queue, task_fused_86_91);
   }
}

template <typename Acc>
void Session<Acc>::infer(std::span<ViewConstF1D const> inputs, std::span<ViewF1D> outputs){
   _infer_impl(inputs[0], inputs[1]);
   alpaka::memcpy(queue, outputs[0], deviceBuf_class_logit);
   alpaka::memcpy(queue, outputs[1], deviceBuf_charge_logit);
   alpaka::memcpy(queue, outputs[2], deviceBuf_reg_pt);
   // alpaka::wait(queue);
}

template <typename Acc>
auto Session<Acc>::infer(BufF1D const deviceBuf_particles,BufF1D const deviceBuf_padding_masks)
   -> std::tuple<ViewF1D,ViewF1D,ViewF1D> {
   ViewConstF1D const view_particles{alpaka::getPtrNative(deviceBuf_particles), devAcc, alpaka::getExtents(deviceBuf_particles)};
   ViewConstF1D const view_padding_masks{alpaka::getPtrNative(deviceBuf_padding_masks), devAcc, alpaka::getExtents(deviceBuf_padding_masks)};
   _infer_impl(view_particles, view_padding_masks);
   alpaka::wait(queue);
   return {deviceBuf_class_logit,deviceBuf_charge_logit,deviceBuf_reg_pt};
}

template <typename Acc>
auto Session<Acc>::infer_nosync(BufF1D const deviceBuf_particles,BufF1D const deviceBuf_padding_masks)
   -> std::tuple<ViewF1D,ViewF1D,ViewF1D> {
   ViewConstF1D const view_particles{alpaka::getPtrNative(deviceBuf_particles), devAcc, alpaka::getExtents(deviceBuf_particles)};
   ViewConstF1D const view_padding_masks{alpaka::getPtrNative(deviceBuf_padding_masks), devAcc, alpaka::getExtents(deviceBuf_padding_masks)};
   _infer_impl(view_particles, view_padding_masks);
   return {deviceBuf_class_logit,deviceBuf_charge_logit,deviceBuf_reg_pt};
}

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
template struct Session<alpaka::TagGpuCudaRt>;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template struct Session<alpaka::TagGpuHipRt>;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
template struct Session<alpaka::TagCpuSerial>;
#endif

} // namespace ...
