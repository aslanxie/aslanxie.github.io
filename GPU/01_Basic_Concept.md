
# 3D 图形流水线和GPGPU


如果需要理解今天GPU, 从3D图形流水线的视角切入, 会是一个好的方式. 如下, 是Wikipedia上[图形流水线](https://en.wikipedia.org/wiki/Graphics_pipeline)介绍中3D图形渲染流水线流程图. 这个流程图看起来比较复杂, 包含了很多不同功能, 但是在这儿我们只是需要抽象的理解: 3D图形流水线的功能非为可编程控制功能, 固定功能, 和接入两者之前的可配置功能.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/3D-Pipeline.svg/1000px-3D-Pipeline.svg.png">
</p>

由于3D图形处理涉及到大量的计算, CPU难以满足其计算需求, 因此显卡(Display/Video Card)扩展了3D图形加速功能. 当3D图形流水线加速功能并入显卡后, 产生了一些本质的变化, 如Peter N. Glaskowsky在[NVIDIA’s Fermi: The First Complete GPU Computing Architecture](https://www.nvidia.com/content/pdf/fermi_white_papers/p.glaskowsky_nvidia%27s_fermi-the_first_complete_gpu_architecture.pdf)GPU历史的总结:
1. 随着芯片制造工艺的提升, 将多芯片构建的3D图形流水线, 集成到了单个芯片.
2. 为了解决程序设计和执行效率问题, 将3D流水线中3个可编程控制功能执行单元(Shader), 汇总到一个统一的执行单元, 共享执行单元, 统一调度.
3. 同时, 通用编程接口的出现, 正式标志这原来的显卡/GPU, 演进成了GPGPU, 正式进入了通用计算领域.

# 生而高效

GPU的3D图形加速引擎, 不论是多芯片时代的独立计算单元, 还是集成单芯片时代的统一计算单元, 都是为了满足图形处理高计算吞吐量的需求. 基于3D图形流水线特性: 大规模的并行数据和简单的数据处理控制逻辑, GPU形成了大规模并行计算单元, 单指令多数据的架构. 即使演进到GPGPU, 依然遵循这些基本的原则. 
在这里GPGPU里的通用, 更多的是相对3D图形计算, 即扩展到图形计算以外的计算, 但依然聚焦于大规模并行数据的计算. 因此其执行流水线的设计相对CPU, 更为精简.

GPU作为计算系统的加速器, 可以不用像CPU一样, 考虑二进制执行程序的兼容性. 因为CPU可以在运行过程中, 动态生成对于GPU的二进制执行程序. 因此, 原则上GPU不需要考虑代际兼容, 可以更进一步简化芯片的设计, 提升芯片的性能.




# Reference
- https://en.wikipedia.org/wiki/Graphics_pipeline

