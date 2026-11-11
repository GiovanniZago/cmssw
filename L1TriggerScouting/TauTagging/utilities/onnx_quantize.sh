#!/bin/bash
QLEVEL="fp16"

# setup python environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh
source /home/gizago/venvs/onnx-trt-env/bin/activate

set -euo pipefail

PYTHON="/home/gizago/venvs/onnx-trt-env/bin/python3"
MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer.bs512.onnx"
QUANTIZED_MODEL="/home/gizago/CMSSW_20_1_0_pre4/src/L1TriggerScouting/JetTagging/data/btagging/model_example_ParticleTransformer_${QLEVEL}.bs512.onnx"

[[ -f "$MODEL" ]] || { echo "Missing input model: $MODEL" >&2; exit 1; }

# TensorRT 11: strongly-typed ONNX; this creates a quantized graph.
# ModelOpt uses random calibration data when calibration_data is not provided.
# Preserve FP32 inputs and outputs for compatibility with CMSSW SoA tensors.
"$PYTHON" - "$MODEL" "$QUANTIZED_MODEL" <<'PY'
import sys

import onnx
from onnxconverter_common import float16

input_path, output_path = sys.argv[1:3]

# Load your existing FP32 ONNX model
model = onnx.load(input_path)

# Convert to FP16
converted = float16.convert_float_to_float16(model)

# Save the new FP16 model
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