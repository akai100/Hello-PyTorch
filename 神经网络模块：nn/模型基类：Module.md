
nn.Module 是所有**神经网络“模块 / 模型”的基类**，负责管理参数、子模块、前向计算和训练状态。

你写的模型本质上就是：

```python3
class MyModel(nn.Module):
    ...
```

## 2. 功能

### 1️⃣ 管理参数（Parameters）

```python3
self.weight = nn.Parameter(...)
```

+ 自动注册为模型参数

+ ```model.parameters()``` 能拿到

+ ```optimizer``` 才能更新它

你 **不需要自己维护参数列表**

### 2️⃣ 管理子模块（Modules）

```python3
self.fc = nn.Linear(10, 3)
```

+ ```fc``` 会被自动注册为子模块

+ ```model.modules()``` / ```model.children()``` 可遍历

+ ```model.to(device)``` 会递归移动

**👉 模块树（Module Tree）自动构建**

### 3️⃣ 统一 forward / call 行为

你只实现：

```python3
def forward(self, x):
    ...
```

调用时：

```python3
y = model(x)
```

实际执行顺序：

```
model(x)
→ __call__()
→ hooks
→ forward(x)
```


**⚠️ 永远不要直接调用 model.forward()**

### 4️⃣ 控制训练 / 推理状态

```python3
model.train()
model.eval()
```

影响哪些模块？

| 模块        | 行为变化                      |
| --------- | ------------------------- |
| Dropout   | 随机 / 关闭                   |
| BatchNorm | 用 batch 统计 / running mean |

+ Dropout

  在训练模式下，Dropout 层会随机丢弃一部分神经元，以减少过拟合。

  在 ```eval()``` 模式下，Dropout 层会停止随机丢弃神经元，转为使用所有神经元，这意味着网络在推理时的表现更加稳定。

+ BatchNorm

  在训练模式下，BatchNorm 会使用每个 mini-batch 的均值和方差来标准化数据。

  而在 ```eval()`` 模式下，BatchNorm 会使用训练时计算的全局均值和方差，而不是每个 batch 的均值和方差。这有助于推理阶段的稳定性，因为推理时通常是基于固定的统计数据
  
📌 ```nn.Module``` 统一管理这个状态

### 5️⃣ 设备 & dtype 统一管理'

```python3
model.cuda()
model.to("cuda")
model.half()
```

### 6️⃣ 保存 & 加载模型

```python3
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

state_dict 本质是：

```
{
  "fc.weight": tensor,
  "fc.bias": tensor,
  ...
}
```
## 内部机制

### 1️⃣ 参数注册原理

```python3
self.w = nn.Parameter(torch.randn(3))
```

内部等价于：

```python3
self._parameters["w"] = Parameter
```

所以：

```
for p in model.parameters():
    ...
```

能遍历到它

### 2️⃣ 子模块注册原理
self.fc = nn.Linear(10, 3)


等价于：

self._modules["fc"] = Linear(...)

### 3️⃣ buffer（非参数但要保存）

```python3
self.register_buffer("mask", torch.ones(10))
```

+ 不参与梯度

+ 会进 ```state_dict```

+ 会随 ```.to(device)```

常见于：BatchNorm、EMA、mask

## 常见问题

### ❌ 1. 忘记 ```super().__init__()```

```python3
class Bad(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 3)  # ❌
```

会导致：

+ 参数不注册

+ optimizer 不更新

### ❌ 2. forward 里创建参数

