
## 1️⃣ 基本概念

```tensor.transpose(dim0, dim1)``` 返回一个**维度交换后的新 tensor，但不拷贝数据**，底层仍然指向同一块内存（类似 view 或 t()）。

+ **参数**

  + dim0：第一个要交换的维度索引

  + dim1：第二个要交换的维度索引

+ **返回值**：一个新的 tensor，维度顺序被交换

+ **特点**：

  + 不改变原始 tensor

  + 通常会改变连续性（contiguous）

  + 如果想在某些操作中使用，需要 .contiguous() 转换

🔑 注意：```tensor.t()``` 是 ```tensor.transpose(0,1)``` 的简化版，只适用于 2D Tensor。
