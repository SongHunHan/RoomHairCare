```python
from glob import glob
import os
```

```python
dataset = sorted(glob('/workspace/ssddata/Hair/kaggleData/dataset/images/*.jpg'))
len(dataset)
```

    95


```python
from sklearn.model_selection import train_test_split

train_data , val_data = train_test_split(dataset,test_size=0.25,random_state=2000)
```


```python
len(train_data) , len(val_data)
```

    (71, 24)


```python
!python train.py --img 640 --batch 8 --epochs 200 --data '/workspace/ssddata/Hair/kaggleData/dataset/data.yaml' --cfg '/workspace/Docker/yolo_skin/yolov5/models/yolov5m.yaml' --weights yolov5m.pt --name hairTets
```


    Overriding model.yaml nc=80 with nc=1
    
                     from  n    params  module                                  arguments                     
      0                -1  1      5280  models.common.Conv                      [3, 48, 6, 2, 2]              
      1                -1  1     41664  models.common.Conv                      [48, 96, 3, 2]                
      2                -1  2     65280  models.common.C3                        [96, 96, 2]                   
      3                -1  1    166272  models.common.Conv                      [96, 192, 3, 2]               
      4                -1  4    444672  models.common.C3                        [192, 192, 4]                 
      5                -1  1    664320  models.common.Conv                      [192, 384, 3, 2]              
      6                -1  6   2512896  models.common.C3                        [384, 384, 6]                 
      7                -1  1   2655744  models.common.Conv                      [384, 768, 3, 2]              
      8                -1  2   4134912  models.common.C3                        [768, 768, 2]                 
      9                -1  1   1476864  models.common.SPPF                      [768, 768, 5]                 
     10                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  2   1182720  models.common.C3                        [768, 384, 2, False]          
     14                -1  1     74112  models.common.Conv                      [384, 192, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  2    296448  models.common.C3                        [384, 192, 2, False]          
     18                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  2   1035264  models.common.C3                        [384, 384, 2, False]          
     21                -1  1   1327872  models.common.Conv                      [384, 384, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  2   4134912  models.common.C3                        [768, 768, 2, False]          
     24      [17, 20, 23]  1     24246  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [192, 384, 768]]
    Model Summary: 369 layers, 20871318 parameters, 20871318 gradients, 48.0 GFLOPs
    

    Starting training for 200 epochs...
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         0/199     3.02G    0.1109   0.02826         0        24       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21    0.00946     0.0476    0.00141   0.000292
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         1/199     3.64G   0.09159   0.02667         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21     0.0176     0.0476    0.00175   0.000327
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         2/199     3.64G   0.08054   0.02794         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21     0.0528     0.0453    0.00356   0.000421
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         3/199     3.64G   0.08441   0.02726         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21     0.0355     0.0476    0.00711    0.00126
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         4/199     3.64G   0.08299   0.02642         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21     0.0174      0.143    0.00852     0.0014
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         5/199     3.64G   0.06703   0.02723         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21     0.0581      0.143     0.0157     0.0049
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         6/199     3.64G   0.06807   0.02962         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.327     0.0476     0.0253    0.00787
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         7/199     3.64G   0.06772   0.02486         0         9       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.147     0.0904     0.0387    0.00641
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         8/199     3.64G    0.0637   0.02646         0        10       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.337       0.19     0.0933      0.017
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
         9/199     3.64G   0.06516   0.03069         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.275     0.0952      0.058     0.0138
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        10/199     3.64G    0.0645   0.02898         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.35      0.238      0.131     0.0283
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        11/199     3.64G   0.05992   0.02467         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.211       0.23      0.113     0.0423
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        12/199     3.64G   0.06112   0.02784         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.218      0.286      0.174     0.0236
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        13/199     3.64G   0.05396   0.02665         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.427      0.286      0.268     0.0474
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        14/199     3.64G   0.04926   0.02819         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.326       0.19      0.147     0.0408
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        15/199     3.64G   0.04759   0.02358         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.359       0.19      0.148     0.0435
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        16/199     3.64G   0.05173   0.02679         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.241      0.381      0.147     0.0424
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        17/199     3.64G   0.05075    0.0274         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.623      0.237      0.253      0.056
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        18/199     3.64G   0.05162   0.02351         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.379      0.523      0.316     0.0888
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        19/199     3.64G   0.04606   0.02627         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.352      0.286      0.173     0.0402
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        20/199     3.64G   0.04551   0.02712         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.249      0.286      0.153     0.0464
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        21/199     3.64G   0.05115   0.02441         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.332      0.476      0.255     0.0934
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        22/199     3.64G   0.04085   0.02507         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.37      0.476      0.257     0.0884
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        23/199     3.64G   0.04398   0.02488         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.322      0.571      0.328     0.0618
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        24/199     3.64G     0.045   0.02699         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.462      0.429      0.315     0.0785
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        25/199     3.64G   0.04774   0.02364         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.55      0.476      0.392      0.101
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        26/199     3.64G   0.04264   0.02271         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.275      0.524      0.232     0.0588
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        27/199     3.64G   0.04039   0.02285         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.418      0.429      0.326     0.0909
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        28/199     3.64G   0.04014   0.02277         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.462      0.613      0.449      0.101
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        29/199     3.64G   0.04744   0.02132         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.458      0.524      0.366     0.0823

                                           ...........
                                           
                                           
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       196/199     3.64G    0.0162   0.00728         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.98      0.573
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       197/199     3.64G   0.01889  0.008089         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.98      0.581
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       198/199     3.64G   0.01532  0.007885         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.983      0.573
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       199/199     3.64G   0.01788  0.008668         0         9       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.984      0.587
    
    200 epochs completed in 0.120 hours.
    Optimizer stripped from runs/train/hairTets11/weights/last.pt, 42.3MB
    Optimizer stripped from runs/train/hairTets11/weights/best.pt, 42.3MB
    
    Validating runs/train/hairTets11/weights/best.pt...
    Fusing layers... 
    Model Summary: 290 layers, 20852934 parameters, 0 gradients, 47.9 GFLOPs
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.904      0.905      0.952      0.636



```python
cd yolov5
```

    /workspace/Docker/yolo_skin/yolov5



```python
!python detect.py --weight /workspace/Docker/yolo_skin/yolov5/runs/train/hairTets11/weights/last.pt --img 640 --conf 0.7 --source /workspace/Docker/yolo_skin/images
```

    YOLOv5 🚀 v6.0-109-g7c6bae0 torch 1.7.1 CUDA:0 (NVIDIA GeForce RTX 3090, 24265MiB)
    
    Fusing layers... 
    Model Summary: 290 layers, 20852934 parameters, 0 gradients, 47.9 GFLOPs
    image 1/7 /workspace/Docker/yolo_skin/images/1637997857357.jpg: 640x480 1 loss, Done. (0.011s)
    image 2/7 /workspace/Docker/yolo_skin/images/20211127_163009.jpg: 480x640 Done. (0.010s)
    image 3/7 /workspace/Docker/yolo_skin/images/20211127_163103.jpg: 640x640 Done. (0.008s)
    image 4/7 /workspace/Docker/yolo_skin/images/5 (1).jpg: 640x608 1 loss, Done. (0.009s)
    image 5/7 /workspace/Docker/yolo_skin/images/5 (11).jpeg: 384x640 1 loss, Done. (0.010s)
    image 6/7 /workspace/Docker/yolo_skin/images/KakaoTalk_20211127_190255322.jpg: 384x640 1 loss, Done. (0.008s)
    image 7/7 /workspace/Docker/yolo_skin/images/KakaoTalk_20211127_190542188.jpg: 384x640 1 loss, Done. (0.008s)
    Speed: 0.2ms pre-process, 9.1ms inference, 0.5ms NMS per image at shape (1, 3, 640, 640)

