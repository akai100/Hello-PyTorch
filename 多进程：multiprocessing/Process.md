```
import torch.multiprocessing as mp

p = mp.Process(target=fn, args=(arg1, arg2, ...), name="my_process")
p.start() # 启动进程
p.join()  # 等待进程结束
```

**核心优势：共享内存 (Shared Memory)**

当你将一个```torch.Tensor``` 传递给```mp.Process``` 的 ```target``` 函数时，PyTorch 不会对该 Tensor 进行拷贝。相反，它会：

+ 将 Tensor 移动到共享内存。

+ 仅向子进程发送一个指向该内存的“句柄”。

+ 任何一个进程对该 Tensor 的原地修改 (In-place operation) 都会反映在所有进程中。
