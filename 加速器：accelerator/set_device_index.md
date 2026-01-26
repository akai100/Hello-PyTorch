```python3
torch.accelerator.set_device_index(device_index)
```

提供了一种**统一的方式来设置当前进程所使用的加速器设备编号**，而无需显式调用 torch.cuda.set_device 或 torch.mps.set_device。

+ device_index

  + 如果传入整数（如 0），它会将当前默认加速器设置为该索引对应的设备

  + 如果传入 ```torch.device``` 对象，它会提取其中的索引进行设置
