import pytest
import torch
import torch.nn as nn
from shrinker.export import export_onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def test_onnx_export_parity(tmp_path):
    model = SimpleModel().eval()
    example_input = torch.randn(1, 10)
    filepath = str(tmp_path / "model.onnx")

    try:
        export_onnx(model, example_input, filepath)
    except ImportError:
        pytest.skip("onnxruntime not installed in environment")

def test_quantized_export_error(tmp_path):
    model = SimpleModel().eval()
    example_input = torch.randn(1, 10)
    filepath = str(tmp_path / "quantized.onnx")

    # simulate quantized parameter dtype correctly without gradient requirements
    with torch.no_grad():
        q_weight = torch.quantize_per_tensor(model.fc.weight.detach(), scale=0.1, zero_point=0, dtype=torch.qint8)
        model.fc.weight = nn.Parameter(q_weight, requires_grad=False)

    with pytest.raises(ValueError, match="quantized parameter dtype"):
        export_onnx(model, example_input, filepath)