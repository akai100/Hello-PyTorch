Dataset = 一个“可索引的数据集合”

它只负责两件事：

1. 给我第 i 个样本

2. 告诉我总共有多少个样本

## 核心接口

自定义 Dataset 只需要实现两个方法：

```python3
class Dataset(object):
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass
```

### ```__getitem__```

+ 按 index 随机访问
+ 返回：
  + ```Tensor```
  + 或 ```(input, target)```
  + 或 ```dict```（推荐）

### '''__len__```

返回数据集大小
