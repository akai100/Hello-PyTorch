## 1️⃣ Leaf Tensor（叶子张量）

在 PyTorch 中，叶子张量是 在计算图的最底层生成的张量，它们 不是由其他张量通过操作产生的。也就是说，它们 没有父节点（grad_fn），或者说它们的 grad_fn 属性为 None。

**特点：**

+ ```requires_grad=True``` 的叶子张量可以在反向传播中累积梯度（grad）

+ 它们通常是我们主动创建的参数或者输入张量

```python3
import torch

# 叶子张量
x = torch.randn(3, 3, requires_grad=True)
print(x.is_leaf)       # True
print(x.grad_fn)       # None
```

这里 ```x``` 是叶子张量，它是直接创建的，```grad_fn=None```，所以它是叶子节点。

## 2️⃣ Non-Leaf Tensor（非叶子张量）

**非叶子张量**是通过 **对叶子张量或其他非叶子张量的操作生成的张量**。它们 **有父节点（grad_fn 不为 None）**，即在计算图中有前驱。

**特点：**

+ 默认情况下，如果 requires_grad=True，非叶子张量在反向传播时**不会保留梯度**，除非你调用 .retain_grad()；

+ 它们是计算图中的中间节点

**示例：**

```python3
y = x + 2
print(y.is_leaf)       # False
print(y.grad_fn)       # <AddBackward0 object at ...>
```

## 3️⃣ 总结对比表

| 特性            | 叶子张量 (Leaf) | 非叶子张量 (Non-Leaf)                 |
| ------------- | ----------- | -------------------------------- |
| `is_leaf`     | True        | False                            |
| `grad_fn`     | None        | 有（AddBackward, MatMulBackward 等） |
| 梯度保存 (`grad`) | 默认保存        | 默认不保存，需要 `.retain_grad()`        |
| 生成方式          | 手动创建或输入数据   | 对叶子或非叶子张量做操作生成                   |
