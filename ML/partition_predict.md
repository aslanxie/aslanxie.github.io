## Partition Predict

The basic process unit of video codec is block. Encoder try different partition, prediction and transform methods on blocks to remove information redundancy and improve compress performance. From history experience, there is a obvious compress performance improvement in every new codec with more advance/complicate algorithm. The increasing of algorithm complexity have exceeded CPU, even GPU, general compute performance. One reason is encoder have to try more options in loop step, including find more optimized partition, prediction and transform methods. Patition method depend on prediction, and prediction method depend on transform. So, more option in every setp means exponential growth compute capacity.

The firt thought to this problem is can I adapt the problem to deep learning. For example, I use CNN to predict partition type. Let's try it. 

## Preparing

To simplify the problem, I try it on AV1 with all key frame, one super blok (fixed size, 64x64) per tile, YUV 4:4:4 input and the first level partition type. Here are aom AV1 encoder command:
```javascript
aom_2.0.1/build/aomenc --passes=2 --pass=1 --width=1920 --height=1080 --bit-depth=8 --i444 --fps=30/1 --target-bitrate=2000 --kf-max-dist=1 --obu --enable-cdef=0 --sb-size=64 --cpu-used=0 --fpf=Lipa64.log -o Lipa64.av1 Lipa_1920x1080.yuv
```
```javascript
aom_2.0.1/build/aomenc --passes=2 --pass=2 --width=1920 --height=1080 --bit-depth=8 --i444 --fps=30/1 --target-bitrate=2000 --kf-max-dist=1 --obu --enable-cdef=0 --sb-size=64 --cpu-used=0 --fpf=Lipa64.log -o Lipa64.av1 Lipa_1920x1080.yuv
```

At the same time, I split frame data to block with partition type. The the video stream is an unbalance data set in 10 partiton types, so I resample the block and keep the balance.

## CNN

I try to build a simple CNN network, the models trained on 800K data set. Here are the model summary:
```javascript
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 62, 62, 128)       3584      
_________________________________________________________________
batch_normalization (BatchNo (None, 62, 62, 128)       512       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 128)       147584    
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 30, 128)       512       
_________________________________________________________________
dropout (Dropout)            (None, 30, 30, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 64)        73792     
_________________________________________________________________
batch_normalization_2 (Batch (None, 28, 28, 64)        256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 64)        36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 13, 13, 64)        256       
_________________________________________________________________
dropout_1 (Dropout)          (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 11, 32)        18464     
_________________________________________________________________
batch_normalization_4 (Batch (None, 11, 11, 32)        128       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 5, 5, 32)          9248      
_________________________________________________________________
batch_normalization_5 (Batch (None, 5, 5, 32)          128       
_________________________________________________________________
flatten (Flatten)            (None, 800)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 800)               0         
_________________________________________________________________
dense (Dense)                (None, 10)                8010      
=================================================================
Total params: 299,402
Trainable params: 298,506
Non-trainable params: 896
_________________________________________________________________
```

The result is not mataching expectation, it only got an accuracy less than 40%, and unbalance on different partition type. 
```javascript
real     [ 87.  96. 121.  94. 113. 102.  96. 101.  93.  98.]
predict  [ 69. 132.  52. 135.  95.  86.  89.  93.  97. 153.]
matching [32. 27. 12. 66. 19. 21. 23. 31. 34. 65.]
```


## RNN

Every seperate block maybe haven't enough feature to detect the type. So, I try it with RNN network to keep whole frame blok as a sequence.
```javascript
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 510, 31, 31, 64)   1792      
_________________________________________________________________
dropout (Dropout)            (None, 510, 31, 31, 64)   0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 510, 15, 15, 64)   36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 510, 15, 15, 64)   0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 510, 7, 7, 64)     36928     
_________________________________________________________________
dropout_2 (Dropout)          (None, 510, 7, 7, 64)     0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 510, 3, 3, 64)     36928     
_________________________________________________________________
dropout_3 (Dropout)          (None, 510, 3, 3, 64)     0         
_________________________________________________________________
reshape (Reshape)            (None, 510, 576)          0         
_________________________________________________________________
lstm (LSTM)                  (None, 510, 128)          360960    
_________________________________________________________________
dropout_4 (Dropout)          (None, 510, 128)          0         
_________________________________________________________________
dense (Dense)                (None, 510, 10)           1290      
=================================================================
Total params: 474,826
Trainable params: 474,826
Non-trainable params: 0
_________________________________________________________________
```

The result looks better, I got near 80% accuracy. Let's check prediction detail in a frame. It is very obviosly, the matached predicon focus on type 0(no partition) and type 3(split partation) 
```javascript
real     [ 56.  22.  21. 367.  12.  10.   8.   5.   7.   2.]
predict  [121.   3.   3. 383.   0.   0.   0.   0.   0.   0.]
matching [ 46.   2.   0. 347.   0.   0.   0.   0.   0.   0.]
```
## Summary

Compared with classical image classfication, a block in frame is usually only a small parts of object, and is difficult to generate enough features to recognise it.
