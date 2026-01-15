
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


📌 ```nn.Module``` 统一管理这个状态

### 5️⃣ 设备 & dtype 统一管理'

```python3
model.cuda()
model.to("cuda")
model.half()
```


