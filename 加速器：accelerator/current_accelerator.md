```python3
torch.accelerator.current_accelerator()
```

以字符串形式返回当前系统检测到的“首选”加速器类型。

返回值详解：

+ "cuda"

  NVIDIA GPU (最常见的加速器)

+ "mps"

  Apple Silicon (M1/M2/M3/M4 系列芯片)

+ "xpu"

  Intel GPU (Data Center Max, Arc 等)

+ "mtia"

  Meta 专有推理加速器

+ "privateuse1"

  第三方自定义硬件后端
