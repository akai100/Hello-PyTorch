## Tensor 和 NumPy 是否共享内存？

PyTorch 的 Tensor 和 NumPy 的 ndarray 可以共享内存，

### 1️⃣ 从 Tensor 转 NumPy

如果你有一个 PyTorch Tensor（在 CPU 上），可以直接用 .numpy() 转成 NumPy 数组：

```python3
import torch
import numpy as np

t = torch.tensor([1, 2, 3], dtype=torch.float32)
n = t.numpy()

# 修改 Tensor 会影响 NumPy
t[0] = 100
print(n)  # [100.   2.   3.]
```

✅ 结论：CPU Tensor 转 NumPy 默认共享内存，不会拷贝数据。

### 2️⃣ 从 NumPy 转 Tensor

如果你有一个 NumPy 数组，可以用 torch.from_numpy() 转成 Tensor：

```python3
n = np.array([4, 5, 6], dtype=np.float32)
t = torch.from_numpy(n)

# 修改 NumPy 会影响 Tensor
n[0] = 400
print(t)  # tensor([400.,   5.,   6.])

```

✅ 结论：NumPy 转 Tensor 也默认共享内存，不会拷贝。

### 3️⃣ 限制

**1. 只限 CPU**：如果 Tensor 在 GPU 上，.numpy() 会报错，需要先 .cpu()。

**2. 数据类型兼容**：NumPy 和 Tensor 的 dtype 要兼容，比如 float32 对应 torch.float32。

**3. 非连续 Tensor**：如果 Tensor 是非连续的（比如经过 transpose），调用 .numpy() 时可能会触发拷贝，而不是共享。
