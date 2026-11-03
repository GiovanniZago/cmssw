"""Package a self-contained TensorRT EPContext + original-ONNX CUDA fallback.

Usage:
  python epcontext_packager.py --model model.onnx --engine model.plan \
      --output model_trt_fallback.onnx --hardware-architecture 9.0 --ep-sdk-version 11.4.0

Alternatively, use --epcontext existing_ctx.onnx instead of --engine, if generated
by ONNX Runtime with ep.context_embed_mode=1. All engine binaries are embedded.

This is a CMSSW-specific metadata convention, not upstream automatic EP fallback.
"""
import argparse
import base64
import hashlib
from pathlib import Path

import onnx
from onnx import external_data_helper, helper

PREFIX = "cmssw.tensorrt_fallback."
KEY_FORMAT = PREFIX + "format"
KEY_MODEL = PREFIX + "onnx_base64"
KEY_SHA256 = PREFIX + "sha256"
FORMAT = "onnx-base64-v1"


def internalize_external_data(model):
    """Ensure the serialized fallback is usable entirely from memory."""
    external_data_helper.convert_model_from_external_data(model)
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError(f"External initializer not internalized: {initializer.name}")
    # Sparse initializer external tensors are uncommon but must be self-contained too.
    for sparse in model.graph.sparse_initializer:
        if sparse.values.data_location == onnx.TensorProto.EXTERNAL or sparse.indices.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError("Sparse initializer contains external data")


def signature(graph):
    return [(x.name, x.type.SerializeToString()) for x in graph.input], [
        (x.name, x.type.SerializeToString()) for x in graph.output
    ]


def attributes(node):
    return {a.name: a for a in node.attribute}


def validate_context(ctx):
    contexts = [n for n in ctx.graph.node if n.domain == "com.microsoft" and n.op_type == "EPContext"]
    if not contexts:
        raise ValueError("EPContext model has no EPContext nodes")
    for node in contexts:
        attrs = attributes(node)
        if attrs.get("source") is not None and attrs["source"].s.decode() != "TensorrtExecutionProvider":
            raise ValueError(f"Unsupported EPContext source in node {node.name!r}")
        if "embed_mode" not in attrs or attrs["embed_mode"].i != 1:
            raise ValueError(f"EPContext node {node.name!r} is not embedded (embed_mode != 1)")
        if "ep_cache_context" not in attrs or not attrs["ep_cache_context"].s:
            raise ValueError(f"EPContext node {node.name!r} has no embedded plan")
    return contexts


def make_manual_context(original, engine_bytes, model_name, hardware_architecture, ep_sdk_version):
    if not engine_bytes:
        raise ValueError("TensorRT engine is empty")
    kwargs = dict(embed_mode=1, ep_cache_context=engine_bytes,
                  source="TensorrtExecutionProvider", main_context=1,
                  partition_name="TensorRT", onnx_model_filename=model_name)
    if hardware_architecture:
        kwargs["hardware_architecture"] = hardware_architecture
    if ep_sdk_version:
        kwargs["ep_sdk_version"] = ep_sdk_version
    node = helper.make_node("EPContext", [x.name for x in original.graph.input],
                            [x.name for x in original.graph.output],
                            name="TensorRT", domain="com.microsoft", **kwargs)
    graph = helper.make_graph([node], "TensorRT_EPContext",
                              list(original.graph.input), list(original.graph.output))
    opsets = [helper.make_opsetid(i.domain, i.version) for i in original.opset_import]
    if not any(i.domain == "com.microsoft" for i in opsets):
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    ctx = helper.make_model(graph, opset_imports=opsets,
                            producer_name="cmssw-tensorrt-packager")
    ctx.ir_version = original.ir_version
    return ctx


def set_metadata(model, key, value):
    if any(p.key == key for p in model.metadata_props):
        raise ValueError(f"Reserved metadata key already present: {key}")
    pair = model.metadata_props.add()
    pair.key = key
    pair.value = value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path,
                   help="Original ONNX graph (the graph CUDA EP must execute)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine", type=Path, help="Engine built offline with trtexec")
    group.add_argument("--epcontext", type=Path,
                       help="Existing ONNX Runtime EPContext with embedded engine(s)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--hardware-architecture", default="", help="e.g. 9.0 or 80+; metadata only")
    p.add_argument("--ep-sdk-version", default="", help="e.g. 11.4.0; metadata only")
    args = p.parse_args()
    if args.output.resolve() in [x.resolve() for x in (args.model, args.engine, args.epcontext) if x]:
        p.error("Output must be different from input files")

    original = onnx.load(str(args.model), load_external_data=True)
    internalize_external_data(original)
    onnx.checker.check_model(original)
    raw_original = original.SerializeToString()

    if args.engine:
        ctx = make_manual_context(original, args.engine.read_bytes(), args.model.name,
                                  args.hardware_architecture, args.ep_sdk_version)
    else:
        ctx = onnx.load(str(args.epcontext), load_external_data=False)
        validate_context(ctx)
        if signature(original.graph) != signature(ctx.graph):
            raise ValueError("EPContext and original ONNX model inputs/outputs differ")

    validate_context(ctx)
    set_metadata(ctx, KEY_FORMAT, FORMAT)
    set_metadata(ctx, KEY_MODEL, base64.b64encode(raw_original).decode("ascii"))
    set_metadata(ctx, KEY_SHA256, hashlib.sha256(raw_original).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(ctx, str(args.output))

    # Inspect the actual serialized artifact, not only the in-memory objects.
    actual = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(actual)
    props = {p.key: p.value for p in actual.metadata_props}
    restored = base64.b64decode(props[KEY_MODEL], validate=True)
    if hashlib.sha256(restored).hexdigest() != props[KEY_SHA256]:
        raise ValueError("The embedded fallback failed the SHA-256 roundtrip check")
    fallback = onnx.load_model_from_string(restored)
    if signature(actual.graph) != signature(fallback.graph):
        raise ValueError("Embedded fallback and EPContext I/O signatures differ")
    print("Wrote:", args.output)
    print("Artifact bytes:", args.output.stat().st_size)
    print("Embedded engine nodes:", len(validate_context(actual)))
    print("Original ONNX bytes:", len(restored))
    print("Fallback SHA-256:", props[KEY_SHA256])
    print("Inputs:", [x.name for x in actual.graph.input])
    print("Outputs:", [x.name for x in actual.graph.output])


if __name__ == "__main__":
    main()