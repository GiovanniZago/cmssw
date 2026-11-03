#!/bin/bash
# Build a self-contained TensorRT EPContext + FP16 CUDA fallback ONNX artifact.
# Requires NVIDIA ModelOpt, ONNX Python package and TensorRT exporter.
EXPORTER="/home/gizago/build/el9_amd64_gcc14/external/tensorrt-exporter/11.4.0.106-6c474231352306affb905f3de94141b4"

set -euo pipefail

: "${EXPORTER:?Set EXPORTER to the TensorRT exporter installation prefix}"
TRTEXEC="${EXPORTER}/bin/trtexec"
PYTHON="/home/gizago/venvs/onnx-trt-env/bin/python3"
SCRIPTS="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/utilities"
MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_fp32_b2048_s16.onnx"
FP16_MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_fp16_b2048_s16.onnx"
ENGINE="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_fp16_b2048_s16.plan"
OUTPUT="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/TauTagging/data/STT_ort_relu_fp16_b2048_s16_fallback.onnx"

[[ -f "$MODEL" ]] || { echo "Missing input model: $MODEL" >&2; exit 1; }
[[ -x "$TRTEXEC" ]] || { echo "Missing executable: $TRTEXEC" >&2; exit 1; }
[[ -f "$SCRIPTS/epcontext_packager.py" ]] || { echo "Missing epcontext_packager.py under $SCRIPTS" >&2; exit 1; }

# TensorRT 11: strongly-typed ONNX; this creates a mixed-FP16 graph.
# ModelOpt uses random calibration data when calibration_data is not provided.
# Preserve FP32 inputs and outputs for compatibility with CMSSW SoA tensors.
"$PYTHON" - "$MODEL" "$FP16_MODEL" <<'PY'
import sys

import onnx
from modelopt.onnx.autocast import convert_to_mixed_precision

input_path, output_path = sys.argv[1:3]

converted = convert_to_mixed_precision(
    onnx_path=input_path,
    low_precision_type="fp16",
    calibration_data=None,
    keep_io_types=True,
)

onnx.checker.check_model(converted)
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

print(f"Mixed-FP16 ONNX model: {output_path}")
print("ONNX input/output names and dtypes preserved.")
PY

# Produce a version-compatible, cross-Ampere+ engine without embedding lean runtime.
LD_LIBRARY_PATH="$EXPORTER/lib:${LD_LIBRARY_PATH:-}" \
"$TRTEXEC" \
  --onnx="$FP16_MODEL" \
  --saveEngine="$ENGINE" \
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
  --model "$FP16_MODEL" \
  --engine "$ENGINE" \
  --output "$OUTPUT" \
  --hardware-architecture '8.9' \
  --ep-sdk-version '11.4.0'
# TODO: restore hardware-architecture 80+
printf '\nCreated artifact: %s\n' "$OUTPUT"
printf 'Use Backend::tensorRT for primary execution and the CMSSW embedded CUDA fallback.\n'
