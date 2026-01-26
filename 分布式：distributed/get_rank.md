```python3
rank = dist.get_rank(group=None)
```

它的作用是：获取当前进程在分布式进程组中的唯一标识符（即 Rank）

+ group: 进程组对象。默认为 None，表示使用由 ```init_process_group``` 创建的全局默认进程组。


## 基本概念

+ Rank (秩)：一个整数，代表当前进程的编号。

+ 范围：从 0 到 world_size - 1。

+ Rank 0：通常被称为**Master Rank**（主进程），负责打印日志、保存模型 Checkpoint 或进行进度条展示。
