"""Builds a minimal valid ONNX model with raw protobuf encoding (NO `onnx` package needed).

The model has input `images` (1,3,H,W) and ONE output that is a fixed constant tensor. That is enough to
test the whole edge runtime (session creation, metadata parsing, letterbox geometry, decoding, NMS) with
a real ONNX Runtime session, without needing trained weights or PyTorch. It says nothing about accuracy.
"""
import struct

import numpy as np


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num, wire, payload):
    return _varint((num << 3) | wire) + payload


def _len(num, data):            # length-delimited
    return _field(num, 2, _varint(len(data)) + data)


def _str(num, s):
    return _len(num, s.encode())


def _int(num, v):               # varint field
    return _field(num, 0, _varint(v))


def _tensor_proto(arr, name=""):
    arr = np.asarray(arr, dtype=np.float32)
    body = b"".join(_int(1, d) for d in arr.shape) + _int(2, 1)      # dims, data_type=FLOAT
    if name:
        body += _str(8, name)
    return body + _len(9, arr.tobytes())                              # raw_data


def _value_info(name, shape):
    dims = b"".join(_len(1, _int(1, d)) for d in shape)               # Dimension{dim_value}
    tensor_type = _int(1, 1) + _len(2, dims)                          # elem_type FLOAT, shape
    return _str(1, name) + _len(2, _len(1, tensor_type))


def build_constant_output_model(output, input_hw=(640, 640), metadata=None):
    """`output`: ndarray returned (unchanged) by every inference. `metadata`: dict -> ONNX metadata_props."""
    output = np.asarray(output, dtype=np.float32)
    attr = _str(1, "value") + _len(5, _tensor_proto(output)) + _int(20, 4)   # type=TENSOR
    node = _str(4, "Constant") + _str(3, "const_out") + _str(2, "output0") + _len(5, attr)
    graph = (_len(1, node) + _str(2, "tiny_yolo_stub")
             + _len(11, _value_info("images", (1, 3) + tuple(input_hw)))
             + _len(12, _value_info("output0", output.shape)))
    model = (_int(1, 8) + _str(2, "karachi-pulse-tests") + _len(7, graph)
             + _len(8, _str(1, "") + _int(2, 13)))                      # ir_version 8, opset 13
    for k, v in (metadata or {}).items():
        model += _len(14, _str(1, k) + _str(2, v))
    return model


def yolo_head(boxes_cxcywh_scores, n_anchors=8400, n_classes=3):
    """Builds a (1, 4+nc, N) YOLOv8-style output with the given detections placed at the first anchors.
    boxes_cxcywh_scores: list of (cx, cy, w, h, class_id, score) in LETTERBOXED pixel coordinates."""
    out = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes_cxcywh_scores):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


# ---------------------------------------------------------------------------- richer stubs (still no `onnx` package)
def _tensor_any(arr, name, dtype):
    """TensorProto for float32 (data_type 1) or int64 (7), with a name (initializers need one)."""
    arr = np.asarray(arr, dtype=np.float32 if dtype == 1 else np.int64)
    body = b"".join(_int(1, d) for d in arr.shape) + _int(2, dtype) + _str(8, name)
    return body + _len(9, arr.tobytes())


def _node(op, inputs, outputs, name, attrs=b""):
    return (b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs)
            + _str(3, name) + _str(4, op) + attrs)


def _attr_ints(name, values):
    return _len(5, _str(1, name) + b"".join(_int(8, v) for v in values) + _int(20, 7))


def _attr_int(name, v):
    return _len(5, _str(1, name) + _int(3, v) + _int(20, 2))


def _value_info_sym(name, dims, elem=1):
    """dims: ints or strings (symbolic)."""
    ds = b"".join(_len(1, _str(2, d) if isinstance(d, str) else _int(1, d)) for d in dims)
    return _str(1, name) + _len(2, _len(1, _int(1, elem) + _len(2, ds)))


def build_conv_detector_model(seed=0, input_hw=(64, 64), n_classes=3, weight_bits=None, metadata=None):
    """A real (tiny, random-weight) conv network with a YOLO-shaped output (1, 4+nc, N):
        images -> Conv3x3/stride4 -> Relu -> Conv1x1 -> Reshape(1, 4+nc, N)
    `weight_bits=8` rounds the weights to a symmetric 8-bit grid and returns the dequantised model - the stand-in for an
    INT8 model whose output differs from FP32 only by weight-rounding error. Used to test the agreement and latency
    machinery without the `onnx` package. It says nothing about detection accuracy."""
    rng = np.random.default_rng(seed)
    h, w = input_hw
    c1, nout = 8, 4 + n_classes
    w1 = rng.normal(0, 0.35, (c1, 3, 3, 3)).astype(np.float32)
    b1 = rng.normal(0, 0.05, c1).astype(np.float32)
    w2 = rng.normal(0, 0.6, (nout, c1, 1, 1)).astype(np.float32)
    b2 = rng.normal(0, 0.05, nout).astype(np.float32)
    if weight_bits:
        def q(a):
            s = np.abs(a).max() / (2 ** (weight_bits - 1) - 1)
            return (np.round(a / s) * s).astype(np.float32)
        w1, w2 = q(w1), q(w2)
    oh, ow = (h + 2 - 3) // 4 + 1, (w + 2 - 3) // 4 + 1
    n = oh * ow
    nodes = [
        _node("Conv", ["images", "w1", "b1"], ["c1"], "conv1",
              _attr_ints("kernel_shape", [3, 3]) + _attr_ints("strides", [4, 4]) + _attr_ints("pads", [1, 1, 1, 1])),
        _node("Relu", ["c1"], ["r1"], "relu1"),
        _node("Conv", ["r1", "w2", "b2"], ["c2"], "conv2", _attr_ints("kernel_shape", [1, 1])),
        _node("Reshape", ["c2", "shape"], ["output0"], "reshape"),
    ]
    graph = b"".join(_len(1, nd) for nd in nodes) + _str(2, "tiny_conv_detector")
    for nm, arr, dt in (("w1", w1, 1), ("b1", b1, 1), ("w2", w2, 1), ("b2", b2, 1),
                        ("shape", np.array([1, nout, n], dtype=np.int64), 7)):
        graph += _len(5, _tensor_any(arr, nm, dt))
    graph += _len(11, _value_info("images", (1, 3, h, w))) + _len(12, _value_info("output0", (1, nout, n)))
    model = (_int(1, 8) + _str(2, "karachi-pulse-tests") + _len(7, graph) + _len(8, _str(1, "") + _int(2, 13)))
    for k, v in (metadata or {}).items():
        model += _len(14, _str(1, k) + _str(2, v))
    return model


def build_gather_encoder_model(table):
    """Stub sentence encoder: Gather(embedding_table, input_ids) -> last_hidden_state (B, L, D). Both `input_ids` and
    `attention_mask` are declared graph inputs, like a real BERT export, so OnnxSentenceEmbedder feeds both."""
    table = np.asarray(table, dtype=np.float32)
    node = _node("Gather", ["table", "input_ids"], ["last_hidden_state"], "gather", _attr_int("axis", 0))
    graph = (_len(1, node) + _str(2, "gather_encoder") + _len(5, _tensor_any(table, "table", 1))
             + _len(11, _value_info_sym("input_ids", ("batch", "seq"), 7))
             + _len(11, _value_info_sym("attention_mask", ("batch", "seq"), 7))
             + _len(12, _value_info_sym("last_hidden_state", ("batch", "seq", table.shape[1]), 1)))
    return _int(1, 8) + _str(2, "karachi-pulse-tests") + _len(7, graph) + _len(8, _str(1, "") + _int(2, 13))
