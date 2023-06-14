import torch
import torchvision
from torch.quantization import QuantStub, DeQuantStub, quantize_static

# Define the model
model = torchvision.models.resnet18(pretrained=True)

# Define the quantization configuration
quant_config = torch.quantization.default_qconfig

# Add quantization and dequantization stubs to the model
model.quant = QuantStub()
model.dequant = DeQuantStub()

# Fuse the modules that are frequently used together
model.eval()
model.fuse_model()

# Quantize the model
quantized_model = quantize_static(model, qconfig=quant_config, dtype=torch.qint8)

# Save the quantized model
torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')