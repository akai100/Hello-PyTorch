## 1️⃣ 基本用法

```Tensor.reshape(*shape)``` 返回一个 新的 ```tensor```，其 数据内容相同，但形状改变为 shape。

```python3
import torch

x = torch.arange(12)  # [0, 1, 2, ..., 11]
print(x.shape)  # torch.Size([12])

y = x.reshape(3, 4)
print(y)
print(y.shape)  # torch.Size([3, 4])
```

## ```-1```的使用

如果你不想手动计算某一维的大小，可以用 -1，PyTorch 会自动计算：

```python3
z = x.reshape(2, -1)  # 自动计算另一维
print(z.shape)  # torch.Size([2, 6])

```

## 与```view```的区别

| 特性   | `view`                | `reshape`                    |
| ---- | --------------------- | ---------------------------- |
| 内存要求 | 必须 **连续（contiguous）** | 不要求连续，会自动复制数据（如果必要）          |
| 返回类型 | tensor 共享内存           | tensor 尝试共享内存，否则会返回新的 tensor |
| 灵活性  | 较严格                   | 更灵活，更安全                      |

**⚠️ 关键点**：如果原 tensor 不是连续的，view 会报错，而 reshape 会自动处理（可能会复制数据）。

```python3
x = torch.arange(6)[::2]  # 不连续
print(x)  # tensor([0, 2, 4])

# x.view(3)  # ❌ 会报错
y = x.reshape(3)  # ✅ 可以成功
print(y)
```
