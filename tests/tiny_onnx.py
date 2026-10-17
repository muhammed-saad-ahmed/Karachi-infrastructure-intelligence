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
