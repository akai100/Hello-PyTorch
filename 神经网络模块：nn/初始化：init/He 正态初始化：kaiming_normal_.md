```python3
def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
```
