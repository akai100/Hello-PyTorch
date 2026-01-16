```python3
# 使用 autocast 在 FP16 下进行前向传播
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
```
