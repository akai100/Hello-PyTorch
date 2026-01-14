```ExponentialLR```是一种**连续软衰减**的学习率调度策略，核心逻辑是**每经过 1 个训练轮数（epoch）或步数（step），就将当前学习率乘以固定的衰减因子 γ（0<γ<1）**—— 和StepLR的 “阶梯式突变衰减” 不同，
它的学习率是平滑、持续下降的，更适合需要缓慢衰减学习率的中小模型 / 小数据集场景。

```python3
ExponentialLR(optimizer: Optimizer,
              gamma: float,
              last_epoch: int = -1,)
```

+ optimizer
+ gamma
+ last_epoch


```ExponentialLR```的学习率衰减遵循指数函数规律，每一轮（或每一步）的学习率计算公式为:

$$lr_{current}=lr_{initial}\times\gamma^t$$

+ $t$：已完成的训练轮数 / 步数（取决于调度器的更新时机）

+ $\gamma$：衰减因子（核心参数，$0<\gamma < 1$），约接近1衰减越慢，越接近0衰减越快
