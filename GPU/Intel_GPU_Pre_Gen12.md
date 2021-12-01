# GPU
[Graphics processing technology has evolved to deliver unique benefits in the world of computing. The latest graphics processing units (GPUs) unlock new possibilities in gaming, content creation, machine learning, and more.](https://www.intel.com/content/www/us/en/products/docs/processors/what-is-a-gpu.html)

## Intel GPU Overview
Before Gen12, Intel provides on-die integrated processor graphics architecture which offers **graphics**, **compute**, **media**, and **display** capabilities. For example, Intel® Core™ i7 processor 6700K is a one-slice instantiation of Intel processor graphics gen9 architecture.

<p align="center">
  <img src="images/Components_Layout_6700K_Gen9.png">
</p>

[The Compute Architecture of Intel® Processor Graphics Gen9, Page 3](https://www.intel.com/content/dam/develop/external/us/en/documents/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0-166010.pdf)

## Intel GPU Block Diagram
Traditionally, GPU contains following components:
- Display/Overlay
- Blitter: Block Image Transferrer, Copy Enine
- GPE: Graphic Processing Engine
- VCE: Video Codec Engine

<p align="center">
  <img src="images/LKF_GPU_Block.png">
</p>
Moving into 3D/Media subsystem
<p align="center">
  <img src="images/Rendering_Engine.png">
</p>

Media pipeline, evolved to general purpose compute pipeline.

[INTEL® UHD GRAPHICS OPEN SOURCE PROGRAMMER'S REFERENCE MANUAL FOR THE 2020 INTEL CORE™ PROCESSORS WITH INTEL HYBRID TECHNOLOGY BASED ON THE "LAKEFIELD" PLATFORM, Volume 3: GPU Overview, Page 1-2](https://01.org/sites/default/files/documentation/intel-gfx-prm-osrc-lkf-vol03-gpu_overview.pdf)

Latest GPU architecture contains:
- Copy/Blitter Engine
- Video Ehancement Engine
- Video Decoder Engine
- Render/Compute Engine
<p align="center">
  <img src="images/command_streamer.png">
</p>

[INTEL® UHD GRAPHICS OPEN SOURCE PROGRAMMER'S REFERENCE MANUAL FOR THE 2020 INTEL CORE™ PROCESSORS WITH INTEL HYBRID TECHNOLOGY BASED ON THE "LAKEFIELD" PLATFORM, Volume 8: Command Stream Programming, Page 1](https://01.org/sites/default/files/documentation/intel-gfx-prm-osrc-lkf-vol08-command_stream_programming.pdf)
