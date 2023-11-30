from ctypes import CDLL, c_bool, c_char_p, c_int, c_ulong, c_void_p
from os.path import exists
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    libann = CDLL("libann.so")
    libann.init.argtypes = c_int, c_int, c_char_p
    libann.init.restype = c_void_p
    libann.load.argtypes = c_void_p, c_char_p, c_char_p, c_char_p, c_bool, c_bool, c_bool, c_char_p
    libann.load.restype = c_int
    libann.embed.argtypes = c_void_p, c_int, c_void_p, c_void_p
    libann.unload.argtypes = c_void_p, c_int
    libann.destroy.argtypes = (c_void_p,)
    libann.shape.argtypes = (c_void_p, c_int, c_bool)
    libann.shape.restype = c_ulong
    libann_available = True
except OSError:
    libann_available = False


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Ann(metaclass=_Singleton):
    def __init__(self, log_level=3, tuning_level=1, tuning_file: str = None) -> None:
        if not libann_available:
            return
        if tuning_file and not exists(tuning_file):
            raise ValueError("tuning_file must point to an existing (possibly empty) file!")
        if tuning_level == 0 and tuning_file is None:
            raise ValueError("tuning_level == 0 reads existing tuning information and requires a tuning_file")
        if tuning_level < 0 or tuning_level > 3:
            raise ValueError("tuning_level must be 0 (load from tuning_file), 1, 2 or 3.")
        if log_level < 0 or log_level > 5:
            raise ValueError("log_level must be 0 (trace), 1 (debug), 2 (info), 3 (warning), 4 (error) or 5 (fatal)")
        self.ann = libann.init(log_level, tuning_level, tuning_file.encode("utf-8") if tuning_file else None)
        self.output_shapes: Dict[int, Tuple[int, ...]] = {}
        self.input_shapes: Dict[int, Tuple[int, ...]] = {}

    def __del__(self) -> None:
        libann.destroy(self.ann)

    def load(
        self,
        model_path: str,
        input_name="input_tensor",
        output_name="output_tensor",
        fast_math=True,
        fp16=False,
        save_cached_network=False,
        cached_network_path: str = None,
    ) -> int:
        if not (exists(model_path) and model_path.endswith((".armnn", ".tflite", ".onnx"))):
            raise ValueError("model_path must be a file with extension .armnn, .tflite or .onnx")
        if cached_network_path and not exists(cached_network_path):
            raise ValueError("cached_network_path must point to an existing (possibly empty) file!")
        if save_cached_network and cached_network_path is None:
            raise ValueError("save_cached_network is True, cached_network_path must be specified!")
        net_id = libann.load(
            self.ann,
            model_path.encode("utf-8"),
            input_name.encode("utf-8"),
            output_name.encode("utf-8"),
            fast_math,
            fp16,
            save_cached_network,
            cached_network_path.encode("utf-8") if cached_network_path else None,
        )

        self.input_shapes[net_id] = self.shape(net_id, input=True)
        self.output_shapes[net_id] = self.shape(net_id, input=False)
        return net_id

    def unload(self, network_id: int) -> None:
        libann.unload(self.ann, network_id)
        del self.output_shapes[network_id]

    def embed(self, network_id: int, input_tensor: NDArray) -> NDArray:
        net_input_shape = self.input_shapes[network_id]
        if input_tensor.shape != net_input_shape:
            raise ValueError(f"input_tensor shape {input_tensor.shape} != network input shape {net_input_shape}")
        output_tensor = np.ndarray(self.output_shapes[network_id], dtype=np.float32)
        libann.embed(
            self.ann, network_id, input_tensor.ctypes.data_as(c_void_p), output_tensor.ctypes.data_as(c_void_p)
        )
        return output_tensor

    def shape(self, network_id: int, input=False) -> Tuple[int]:
        s = libann.shape(self.ann, network_id, input)
        a = []
        while s != 0:
            a.append(s & 0xFFFF)
            s >>= 16
        return tuple(a)
