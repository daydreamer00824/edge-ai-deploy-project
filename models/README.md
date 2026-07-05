
# Models

本目录用于存放 ONNX 模型和 TensorRT engine 文件。

由于模型文件和 TensorRT engine 文件体积较大，且 TensorRT engine 与 GPU / CUDA / TensorRT 版本相关，本仓库默认不提交模型文件。

请按照根目录 README 中的命令自行生成：

- model.onnx
- model_fp32.engine
- model_fp16.engine
