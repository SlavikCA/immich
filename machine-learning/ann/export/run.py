import logging
import os
import platform
import subprocess

import open_clip
import torch
from tinynn.converter import TFLiteConverter
from onnx2torch import convert
import onnx
from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed, fix_output_shapes

class ExportBase(torch.nn.Module):
    def __init__(self, device: torch.device, name: str):
        super().__init__()
        self.device = device
        self.name = name
    
    def dummy_input(self):
        pass

class ArcFace(ExportBase):
    def __init__(self, onnx_model_path: str, device: torch.device):
        name, _ = os.path.splitext(os.path.basename(onnx_model_path))
        super().__init__(device, name)
        self.input_shape = (1,3,112,112)
        onnx_model = onnx.load_model(onnx_model_path)
        make_input_shape_fixed(onnx_model.graph, onnx_model.graph.input[0].name, self.input_shape)
        fix_output_shapes(onnx_model)
        self.model = convert(onnx_model).to(device)
        if self.device.type == "cuda":
            self.model = self.model.half()

    def forward(self, input_tensor):
        embedding = self.model(input_tensor.half() if self.device.type == "cuda" else input_tensor)
        return embedding.float()
    
    def dummy_input(self):
        return torch.rand(self.input_shape, device=self.device)


class ClipVision(ExportBase):
    def __init__(self, device: torch.device, model_name: str = "ViT-B-32", weights: str = "openai"):
        super().__init__(device, model_name + "__" + weights)
        self.model = open_clip.create_model(
            model_name,
            weights,
            precision="fp16" if device.type == "cuda" else "fp32",
            jit=False,
            require_pretrained=True,
            device=device,
        )

    def forward(self, input_tensor: torch.FloatTensor):
        embedding = self.model.encode_image(input_tensor.half() if self.device.type == "cuda" else input_tensor)
        return embedding.float()
    
    def dummy_input(self):
        return torch.rand((1, 3, 224, 224), device=self.device)

def export(model: ExportBase):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    dummy_input = model.dummy_input()
    dummy_out = model(dummy_input)
    jit = torch.jit.trace(model, dummy_input)
    output_name = "output_tensor"
    list(jit.graph.outputs())[0].setDebugName(output_name)
    tflite_model_path = f"output/{model.name}.tflite"
    os.makedirs("output", exist_ok=True)

    converter = TFLiteConverter(jit, dummy_input, tflite_model_path, nchw_transpose=True)
    # segfaults on ARM, must run on x86_64 / AMD64
    converter.convert()

    armnn_model_path = f"output/{model.name}.armnn"
    os.environ.LD_LIBRARY_PATH = "armnn"
    subprocess.run(
        [
            "./armnnconverter",
            "-f",
            "tflite-binary",
            "-m",
            tflite_model_path,
            "-i",
            "input_tensor",
            "-o",
            "output_tensor",
            "-p",
            armnn_model_path,
        ]
    )

def main():
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(f"Can only run on x86_64 / AMD64, not {platform.machine()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning(
            "No CUDA available, cannot create fp16 model! " "proceeding to create a fp32 model (use only for testing)"
        )
    onnx_model_path = 'buffalo_l.onnx'
    models = [
        ClipVision(device),
        ArcFace(onnx_model_path, device),
    ]
    for model in models:
        export(model)
    


if __name__ == "__main__":
    with torch.no_grad():
        main()
