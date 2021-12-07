# Quantization
PyTorch introduces three models for quantization:
- Dynamic Quantization, dynamic quantization (weights quantized with activations read/stored in floating point and quantized for compute.)
- Post-Training Static Quantization, static quantization (weights quantized, activations quantized, calibration required post training)
- Quantization Aware Training, static quantization aware training (weights quantized, activations quantized, quantization numerics modeled during training)

[Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
