```python3
scaler = GradScaler()

# 使用 GradScaler 进行梯度缩放和反向传播
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
