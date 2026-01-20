```python3
class Embedding(Module):
    ....
```

```torch.nn.Embedding``` 是深度学习（尤其是 NLP）中非常核心的一个层。你可以把它想象成一个查找表 (Lookup Table)，它的作用是将离散的整数索引（比如单词的 ID）映射成连续的稠密向量（Embedding Vector）。

## 初始化

```python3
def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
```

+ num_embeddings

  词表的大小。比如你有 1000 个单词，这里就填 1000

+ embedding_dim

  每个单词对应的向量维度。比如你想用 128 维的向量来表示一个词。

+ padding_idx

  填充索引。如果你指定了它，那么这个索引对应的向量在初始化时会被设为 0，并且在训练中不会更新权重。

+ max_norm

  如果指定，任何范数超过此值的向量都会被归一化到此值。

+ norm_type

  计算 max_norm 时使用的 p-范数

+ scale_grad_by_freq

  是否根据词在 batch 中出现的频率缩放梯度。

+ sparse

  如果为 True，权重 W 的梯度将是一个稀疏张量（有助于处理超大词表）。
