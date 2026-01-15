## 1️⃣ 基本概念

在 PyTorch 中，每个 ```Tensor``` 都有一个属性 ```requires_grad```：

+ 如果 ```requires_grad=True```，这个 ```tensor``` 就会记录 计算图 上的操作，以便后续求梯度。

+ 这个计算图是一个 ```有向无环图 (DAG)```，节点是 tensor，边是操作。

## 2️⃣ backward() 做的事

假设你有一个 scalar（标量）```loss```：

```python3
loss.backward()
````

```backward()``` 的作用可以分解为三个步骤：

**(1) 从输出开始构建梯度图**

+ 从你调用 ```backward()``` 的 ```tensor``` 出发（通常是损失 loss），沿着计算图反向遍历所有操作。

+ 每个节点保存了如何 **根据它的输入计算梯度** 的信息（也就是链式法则）

**(2) 链式法则传播梯度**

+ PyTorch 使用 **链式法则 (Chain Rule)** 从输出向输入传播梯度。

对于每个 tensor 

$$x.grad = \frac{∂loss}{∂x}$$

+ 这个梯度会累加到 $x.grad$ 中，如果你没有手动清零的话

## 3️⃣ 关键细节

**1. 只能对标量调用** ```backward()```

  + 如果是非标量，需要传入 gradient 参数：
  ```
    y.backward(torch.ones_like(y))
  ```

**2. 累加梯度**

  + ```.grad``` 默认是累加的，所以训练前通常要 ```optimizer.zero_grad()```。

**3. 非叶子 tensor 默认不保存 grad**

  + 可以通过 ```retain_grad()``` 来让非叶子节点也保存梯度：

```python3
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * 2
y.retain_grad()
z = y.sum()
z.backward()
print(y.grad)  # 可以访问
````


**4. in-place 操作可能破坏计算图**

  + 如果你对计算图中的 ```tensor``` 做了 in-place 操作，```backward()``` 可能报错或得到错误梯度。

## 4. 参数

### 4.1 retain_graph

当你需要对同一份计算图调用 ```backward()``` 多次时，才需要 ```retain_graph=True```

#### 4.1.1 为什么默认不能多次 backward？

PyTorch 的设计是：

+ **反向传播一次后**

+ 计算图会被释放（free）

+ 节省显存

```python3
y = f(x)
y.backward()   # OK
y.backward()   # ❌ RuntimeError
```

报错本质是：backward 后，计算图已经被销毁

#### 4.1.2 ```retain_graph=True``` 是干嘛的？

```python3
y.backward(retain_graph=True)
```

含义是：**反向传播后，不释放计算图**

这样你可以 再 **backward 一次**
