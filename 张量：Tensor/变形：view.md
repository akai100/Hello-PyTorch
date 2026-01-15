
不在改变数据本身的情况下，改变 Tensor 的形状（shape）。

## 基本用法

```python3
import torch

x = torch.arange(12)
print(x)            # shape:[12]

y = x.view(3, 4)
print(y)            # shape:[3, 4]
```

## -1：自动推断维护

```python3
x = torch.arange(24)

x1 = x.view(2, -1)     # 自动算出第二维
x2 = v.view(-1, 6)     # 自动算出第一维
```

## 最重要的坑：```view()```要求连续内存

**什么是连续（contiguous）?**

```python3
x = torch.randn(2, 3)
x.is_contiguous()    # True
```

但以下操作会破快连续性：

```python3
y = x.t()            # 转置
y.is_contiguous()    # False
```

此时：

```python3
y.view(6)            # ❌ RuntimeError
```

✅ 正确写法

```python3
y.contiguous().view(6)
```

## 5️⃣ view() vs reshape()（高频面试点）

| 方法          | 是否要求连续 | 是否复制数据  |
| ----------- | ------ | ------- |
| `view()`    | ✅ 必须连续 | ❌ 不复制   |
| `reshape()` | ❌ 不要求  | ⚠️ 可能复制 |

```python3
x.reshape(6)   # 更安全
x.view(6)      # 更快但有限制
```

**👉 工程中推荐：** reshape()

**性能敏感 & 确定连续**：view()

## 6️⃣ CNN / Transformer 中的典型用法

### Flatten（展开）

```python3
x = torch.randn(32, 128, 7, 7)

x = x.view(x.size(0), -1)
# [batch_size, features]
```

### Attention 中拆分 head

```python3
x = x.view(batch, seq_len, num_heads, head_dim)
```

## 7️⃣ view() 的本质（一句话）

```view()``` 不动数据，只改 ```shape```；前提是内存布局不能变
