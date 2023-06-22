import torch
import torch.utils.data
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models

sys.path.append("path to torchvision/references/classification/")
from train import evaluate, train_one_epoch, load_data