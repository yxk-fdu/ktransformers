# Ascend CPU MoE 后端

配套说明见 [CPU MoE 使用指南](https://github.com/yxk-fdu/vllm-ascend/blob/share/cpu-moe-9334-20261020/docs/cpu_moe/README.zh.md)。

本分支包含 Ascend 适配、原生 MXFP4、GGUF、异步提交/完成及 CPU 计算日志源码。运行包 Python 文件来自 9334 的 kt_async_runtime；C++ 使用当前源码树，包含空 CPU 路由跳过计算的实现，不引入实验目录中的诊断消融版本。

先准备匹配的 torch/torch_npu/CANN 环境，以及 hwloc-devel（Debian 为 libhwloc-dev）、pkg-config、CMake、GCC/G++、pybind11、numpy、safetensors、gguf。

```bash
git submodule update --init third_party/llama.cpp third_party/pybind11
git -C third_party/llama.cpp apply ../../patches/llama-cpp-mxfp4.patch
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash kt-kernel/build_ascend_cpu_moe.sh /your/work/kt-build
export PYTHONPATH=/your/work/kt-build/package:${PYTHONPATH:-}
```

补丁对应 llama.cpp `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`。不要重复应用。转换器 `tools/convert_mxfp4_layer_to_gguf.py` 使用该依赖的 gguf-py。产物不提交到仓库；目标机器必须按自身 CPU、Python ABI 和 Ascend 环境重新编译。

9334 使用 Python 3.11.10、torch 2.10.0+cpu、torch_npu 2.10.0.post2。旧 torch<2.9 上限已放宽到 <2.11；安装时仍需显式准备匹配版本。独立构建脚本不运行 pip 依赖解析，不会自动替换现有 torch。纯 Ascend 构建不需要 CUDA Triton。

9334 GGUF 可复用 `/mnt/sdc/DeepSeek-V4-Flash-0731-GGUF`；原生模型为 `/mnt/data/DeepSeek-V4-Flash-0731`。其他服务器先问作者是否已有权重。
