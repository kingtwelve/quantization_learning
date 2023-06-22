# Quantization Learning
## pytorch 学习资料
https://pytorch.org/blog/introduction-to-quantization-on-pytorch/
https://pytorch.org/docs/stable/quantization.html#quantization-flow
https://pytorch.org/docs/stable/quantization.html#quantization-api-summary
https://pytorch.apachecn.org/1.7/57/#5
## 注意事项
只能在cpu上进行量化

CNN没有办法是用那个dynamic quantization，只能使用static quantization
## Learning 1 -- Post Training Quantization
是按照pytorch给出的官方教程[(BETA) STATIC QUANTIZATION WITH EAGER MODE IN PYTORCH](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization)
来学习的，使用的模型是MobileNetV2，数据集是imageNet

## Learning 2 -- 
根据NVIDIA给出的tensorRT的教程来写的

## Learning 3 -- 
根据 [https://pytorch.org/TensorRT/tutorials/ptq.html](https://pytorch.org/TensorRT/tutorials/ptq.html)来写
## 一些问题
### 模型定义中_make_divisable的作用？可不可以不使用
### 模型校准torch.quantization.prepare和torch.quantization.convert中间的部分作用
