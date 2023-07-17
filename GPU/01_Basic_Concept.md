
# 3D 图形流水线和GPGPU


如果需要理解今天GPU, 从3D图形流水线的视角切入, 会是一个好的方式. 如下图, 是Wikipedia上[图形流水线](https://en.wikipedia.org/wiki/Graphics_pipeline)介绍中3D图形渲染流水线流程图. 这个流程图看起来比较复杂, 包含了很多不同功能, 但是在这儿我们只是需要抽象的理解: 3D图形流水线的功能可以划分为可编程控制功能, 固定功能, 和介于两者之前的可配置功能.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/3D-Pipeline.svg/1000px-3D-Pipeline.svg.png">
</p>

由于3D图形处理涉及到大量的计算, CPU难以满足其计算需求, 因此显卡(Display/Video Card)扩展了3D图形加速功能. 当3D图形流水线加速功能并入后, 显卡产生了一些本质的变化, 如Peter N. Glaskowsky在 [NVIDIA’s Fermi: The First Complete GPU Computing Architecture](https://www.nvidia.com/content/pdf/fermi_white_papers/p.glaskowsky_nvidia%27s_fermi-the_first_complete_gpu_architecture.pdf) 回顾GPU历史时的总结:
1. 随着芯片制造工艺的提升, 将多芯片构建的3D图形流水线, 集成到了单个芯片GPU.
2. 为了解决程序设计和执行效率问题, 将3D流水线中3个可编程控制功能执行单元(Shader), 抽象到一个统一的执行引擎, 共享执行单元, 统一调度.

因此, 从芯片架构层面, GPU已经演变成了的计算处理器. 从软件层面, Pat Hanrahan在\<<OpenCL Programming Guide\>>序言中提到:"Computing systems are a symbiotic combination of hardware and software. Hardware is not useful without a good programming model.", 并回顾了早期从Shading Language到CUDA和OpenCL的一些历史. 通用编程接口的出现, 正式标志这原来的显卡, 演进成了GPGPU, 源自于图形计算, 又超越了图形计算的范围, 实现更为通用的计算.

# 生而高效

构建一个计算处理器芯片所需要的功能, 远超单纯的计算单元或者引擎. 在[\<<In-Datacenter Performance Analysis of a Tensor Processing Unit\>>](https://arxiv.org/ftp/arxiv/papers/1704/1704.04760.pdf) 中, 详细列出的Google TPU中主要功能占用芯片的面积, 其中data buffers占芯片的37%, compute占30%. 并概述了其设计理念: 极简主义是特定领域处理器的优点. 因此没有采用复杂的架构特性,如分支预测, 乱序执行等等, 来提升平均性能, 这些高级特性往意味着更多的晶体管和功率消耗.

同CPU相比, GPGPU也一样, 源自于3D图形流水线专用计算引擎, 其通用是参考原始的图形处理功能, 而不是对比CPU的通用. 因此在架构设计上, 依然避免了分支预测, 乱序执行等一下CPU架构中的复杂功能, 而专注于单指令多数据, 以及与计算匹配的缓存系统, 提升计算所占晶体管和功耗比例, 从而实现更高的计算吞吐量.


另一方面, GPU作为协处理器, 其运行程序依赖于CPU实时编译和加载, 因此GPU并不需要考虑代际兼容, CPU会协助将GPU上执行程序编译成对应GPU所需要的格式. 因此, GPU的架构相对CPU可以更为精简, 更加专注于计算性能.



# Reference
- https://en.wikipedia.org/wiki/Graphics_pipeline

