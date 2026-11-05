#!/bin/bash
# Build a self-contained TensorRT EPContext + FP16 CUDA fallback ONNX artifact.
# Requires NVIDIA ModelOpt, ONNX Python package and TensorRT exporter.
EXPORTER="/home/gizago/build/el9_amd64_gcc14/external/tensorrt-exporter/11.4.0.106-6c474231352306affb905f3de94141b4"
QLEVEL="int8"

# setup python environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh
source /home/gizago/venvs/onnx-trt-env/bin/activate

set -euo pipefail

: "${EXPORTER:?Set EXPORTER to the TensorRT exporter installation prefix}"
TRTEXEC="${EXPORTER}/bin/trtexec"
PYTHON="/home/gizago/venvs/onnx-trt-env/bin/python3"
SCRIPTS="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/utilities"
# MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_fp32_b2048_s16.onnx"
# QUANTIZED_MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_${QLEVEL}_b2048_s16.onnx"
# ENGINE="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_${QLEVEL}_b2048_s16.plan"
# OUTPUT="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_${QLEVEL}_b2048_s16_fallback.onnx"
MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer.bs512.onnx"
QUANTIZED_MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer.bs512.${QLEVEL}.onnx"
ENGINE="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer.bs512.${QLEVEL}.plan"
OUTPUT="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer.bs512.${QLEVEL}.fallback.onnx"

[[ -f "$MODEL" ]] || { echo "Missing input model: $MODEL" >&2; exit 1; }
[[ -x "$TRTEXEC" ]] || { echo "Missing executable: $TRTEXEC" >&2; exit 1; }
[[ -f "$SCRIPTS/epcontext_packager.py" ]] || { echo "Missing epcontext_packager.py under $SCRIPTS" >&2; exit 1; }

# TensorRT 11: strongly-typed ONNX; this creates a quantized graph.
# ModelOpt uses random calibration data when calibration_data is not provided.
# Preserve FP32 inputs and outputs for compatibility with CMSSW SoA tensors.
"$PYTHON" - "$MODEL" "$QUANTIZED_MODEL" <<'PY'
import sys

import onnx
from modelopt.onnx.quantization import quantize

input_path, output_path = sys.argv[1:3]

quantize(
    onnx_path=input_path,
    output_path=output_path,
    quantize_mode="int8",
    calibration_data=None,
    calibration_method="max",
    calibration_eps=["cpu"],
    high_precision_dtype="fp32",
    direct_io_types=False,
)

converted = onnx.load(output_path)
onnx.save(converted, output_path)

# Verify that ModelOpt preserved the original ONNX input/output interface.
original = onnx.load(input_path)

for kind in ("input", "output"):
    before = [
    (value.name, value.type.tensor_type.elem_type)
        for value in getattr(original.graph, kind)
    ]
    after = [
    (value.name, value.type.tensor_type.elem_type)
        for value in getattr(converted.graph, kind)
    ]

    if before != after:
        raise SystemExit(
            f"{kind} interface changed:\n"
            f"  Original:  {before}\n"
            f"  Converted: {after}"
        )

print(f"Quantized ONNX model: {output_path}")
print("ONNX input/output names and dtypes preserved.")
PY

# Produce a version-compatible, cross-Ampere+ engine without embedding lean runtime.
# LD_LIBRARY_PATH="$EXPORTER/lib:${LD_LIBRARY_PATH:-}" \
# "$TRTEXEC" \
#   --onnx="$QUANTIZED_MODEL" \
#   --saveEngine="$ENGINE" \
#   --versionCompatible \
#   --excludeLeanRuntime \
#   --profilingVerbosity=detailed \
#   --skipInference

JET_SHAPES="pf_features:512x15x16,pf_vectors:512x4x16,pf_mask:512x1x16"

LD_LIBRARY_PATH="$EXPORTER/lib:${LD_LIBRARY_PATH:-}" \
"$TRTEXEC" \
  --onnx="$QUANTIZED_MODEL" \
  --saveEngine="$ENGINE" \
  --minShapes="$JET_SHAPES" \
  --optShapes="$JET_SHAPES" \
  --maxShapes="$JET_SHAPES" \
  --versionCompatible \
  --excludeLeanRuntime \
  --profilingVerbosity=detailed \
  --skipInference
# TODO: add again --hardwareCompatibilityLevel=ampere+ \
#  --minShapes=particles:1x16x10,padding_masks:1x1 \
#  --optShapes=particles:2048x16x10,padding_masks:2048x1 \
#  --maxShapes=particles:16384x16x10,padding_masks:16384x1 \
# The packager embeds the plan and base64-encoded original FP16 graph in one ONNX.
"$PYTHON" "$SCRIPTS/epcontext_packager.py" \
  --model "$QUANTIZED_MODEL" \
  --engine "$ENGINE" \
  --output "$OUTPUT" \
  --hardware-architecture '8.9' \
  --ep-sdk-version '11.4.0'
# TODO: restore hardware-architecture 80+
printf '\nCreated artifact: %s\n' "$OUTPUT"
printf 'Use Backend::tensorRT for primary execution and the CMSSW embedded CUDA fallback.\n'
