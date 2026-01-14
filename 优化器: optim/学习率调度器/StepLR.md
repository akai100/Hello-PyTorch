
**按固定步数/轮数间隔，将学习率乘以衰减因子进行阶梯式下降**，属于硬衰减策略，常用于中小模型或结构化数据的训练场景。

```python3
torch.optim.lr_scheduler.StepLR(optimizer: Optimizer,
                                step_size: int,
                                gamma: float = 0.1,
                                last_epoch: int = -1)
```

+ optimizer

  关联的优化器实例

+ step_size

  每经过```step_size```个单位，就将当前学习率乘以```gamma```（衰减因子），公式如下：

$$lr_{current}=lr_{initial}\times\gamma^{\frac{epoch}{step\\_size}}$$

+ gamma

  学习率衰减因子，0.1（常用）、0.5

+ last_epoch

  上一次更新的轮数，用于断点续训，-1（默认，从头开始）

## StepLR 是什么？

**StepLR 是一种「分段式学习率衰减（step decay）」策略**

一句话：

**每训练固定步数，就把学习率乘一个常数**

## 为什么要用 StepLR？（核心动机）

### 1️⃣ 固定学习率的问题

+ 初期：

  + 学习率太小 → 学得慢

+ 后期：

  + 学习率太大 → 在最优点附近震荡

👉 一个学习率不可能全程最优

### 2️⃣ StepLR 的直觉

前期：大步探索

后期：小步精修

而 StepLR 用的是：

+ 最简单

+ 最稳定

+ 最可控

的方式来实现这一点

## 数学形式

设：

+ 初始学习率： $\eta_0$
	​
+ 每```step_size```衰减一次

+ 衰减因子： $\gamma \in (0, 1)$

**第 $k$次衰减后的学习率：**

$$\eta_k=\eta_0 \cdot \gamma^k$$
