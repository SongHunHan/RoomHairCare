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

    [34m[1mtrain: [0mweights=yolov5m.pt, cfg=/workspace/Docker/yolo_skin/yolov5/models/yolov5m.yaml, data=/workspace/ssddata/Hair/kaggleData/dataset/data.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=200, batch_size=8, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, name=hairTets, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=0, save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
    [34m[1mgithub: [0mskipping check (Docker image), for updates see https://github.com/ultralytics/yolov5
    YOLOv5 🚀 v6.0-109-g7c6bae0 torch 1.7.1 CUDA:0 (NVIDIA GeForce RTX 3090, 24265MiB)
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mWeights & Biases: [0mrun 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/train', view at http://localhost:6006/
    Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt to yolov5m.pt...
    100%|██████████████████████████████████████| 40.7M/40.7M [00:03<00:00, 10.7MB/s]
    
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
    
    Transferred 474/481 items from yolov5m.pt
    Scaled weight_decay = 0.0005
    [34m[1moptimizer:[0m SGD with parameter groups 79 weight, 82 weight (no decay), 82 bias
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mval: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/val.cache' images and [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mval: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/val.cache' images and [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    [34m[1mval: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/val.cache' images and [0m
    [34m[1mval: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/val.cache' images and [0m
    [34m[1mtrain: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/train.cache' images [0m
    Plotting labels to runs/train/hairTets11/labels.jpg... 
    [34m[1mval: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/val.cache' images and [0m
    [34m[1mval: [0mScanning '/workspace/ssddata/Hair/kaggleData/dataset/val.cache' images and [0m
    
    [34m[1mAutoAnchor: [0m3.16 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
    Image sizes 640 train, 640 val
    Using 8 dataloader workers
    Logging results to [1mruns/train/hairTets11[0m
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
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        30/199     3.64G   0.04293   0.01875         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.418      0.429      0.336      0.129
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        31/199     3.64G   0.04239   0.02112         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.3      0.762      0.319      0.127
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        32/199     3.64G   0.04188   0.02188         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.444      0.571      0.406      0.123
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        33/199     3.64G   0.04826    0.0237         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.333      0.619      0.355      0.154
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        34/199     3.64G   0.04896   0.02486         0         8       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.473      0.429      0.354      0.135
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        35/199     3.64G   0.05197   0.02123         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.625      0.476      0.508      0.172
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        36/199     3.64G   0.03959   0.02006         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.521      0.571      0.535      0.147
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        37/199     3.64G   0.03811   0.02348         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.523      0.524      0.459      0.145
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        38/199     3.64G   0.04126   0.02095         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.666      0.476      0.505       0.14
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        39/199     3.64G   0.03969   0.02194         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.733      0.524      0.537      0.125
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        40/199     3.64G   0.04549   0.02164         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.833      0.475      0.608      0.207
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        41/199     3.64G   0.03993   0.01764         0         8       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.588      0.476      0.473      0.188
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        42/199     3.64G   0.04213    0.0206         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.846      0.522      0.643      0.227
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        43/199     3.64G   0.04872   0.02034         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.218      0.571      0.321     0.0825
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        44/199     3.64G   0.04291   0.01671         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.391      0.524      0.359     0.0837
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        45/199     3.64G   0.04165   0.01708         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.496      0.619      0.531      0.201
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        46/199     3.64G   0.03528   0.02172         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.857      0.569      0.723       0.31
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        47/199     3.64G   0.04163   0.02028         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.553      0.476      0.554      0.229
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        48/199     3.64G   0.03623   0.01638         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.464      0.619      0.596      0.278
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        49/199     3.64G    0.0518   0.01943         0        10       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.811      0.619      0.708      0.247
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        50/199     3.64G     0.043   0.01939         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.875      0.666      0.812      0.344
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        51/199     3.64G   0.03888   0.01832         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.748      0.566      0.711      0.326
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        52/199     3.64G   0.04344   0.01972         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.483      0.714      0.617      0.279
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        53/199     3.64G   0.04465   0.01775         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.607      0.808      0.705      0.297
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        54/199     3.64G     0.036   0.01736         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.632      0.571      0.604      0.277
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        55/199     3.64G    0.0359   0.01699         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.702      0.571       0.63      0.297
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        56/199     3.64G   0.04522    0.0188         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.788       0.71       0.79      0.349
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        57/199     3.64G   0.04634   0.01732         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.928      0.619      0.753      0.273
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        58/199     3.64G   0.03926   0.01492         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.748      0.714      0.789      0.389
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        59/199     3.64G   0.03975   0.01753         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.933      0.666      0.817      0.337
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        60/199     3.64G   0.03477   0.01663         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.498      0.905      0.631      0.311
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        61/199     3.64G   0.04103   0.01517         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.536      0.714      0.671      0.361
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        62/199     3.64G   0.04111   0.01902         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.714      0.714      0.702      0.341
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        63/199     3.64G   0.03795   0.01662         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.765      0.619       0.75      0.329
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        64/199     3.64G   0.03837   0.01795         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.636      0.666      0.676      0.329
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        65/199     3.64G     0.035   0.01386         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.583      0.667      0.602      0.238
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        66/199     3.64G    0.0379   0.01705         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.75      0.714      0.755      0.323
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        67/199     3.64G   0.03691   0.01731         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.866      0.619      0.776      0.355
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        68/199     3.64G   0.03816   0.01587         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.933      0.666      0.851      0.391
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        69/199     3.64G    0.0293   0.01359         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.847       0.81      0.817      0.359
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        70/199     3.64G   0.03936   0.01688         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.889      0.761      0.879      0.382
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        71/199     3.64G    0.0371   0.01396         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.791      0.905      0.882      0.425
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        72/199     3.64G   0.04019   0.01515         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.882      0.714      0.827      0.271
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        73/199     3.64G   0.04055   0.01344         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.543      0.905      0.683      0.315
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        74/199     3.64G   0.04229   0.01502         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.705      0.571      0.624      0.319
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        75/199     3.64G   0.03717   0.01825         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.636      0.666      0.723      0.362
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        76/199     3.64G   0.04016   0.01675         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.782      0.857      0.899      0.408
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        77/199     3.64G   0.04145   0.01482         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.79      0.905      0.912      0.375
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        78/199     3.64G    0.0364   0.01651         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.842      0.762      0.885      0.459
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        79/199     3.64G   0.03893   0.01545         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.9      0.855      0.915      0.426
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        80/199     3.64G   0.03522   0.01365         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.593      0.762       0.72      0.359
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        81/199     3.64G   0.03505   0.01657         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.809       0.81      0.855      0.472
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        82/199     3.64G   0.02526   0.01368         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.666      0.835      0.471
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        83/199     3.64G   0.03314   0.01356         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.667      0.824      0.431
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        84/199     3.64G   0.03152   0.01429         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.536      0.714      0.608      0.295
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        85/199     3.64G   0.03094   0.01207         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.5      0.761      0.588      0.318
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        86/199     3.64G   0.03887   0.01377         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.619      0.779      0.427
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        87/199     3.64G   0.03144   0.01361         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.933      0.667       0.79      0.388
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        88/199     3.64G   0.03174   0.01406         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.667      0.804      0.388
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        89/199     3.64G   0.03738   0.01602         0        21       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.577      0.714      0.678      0.324
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        90/199     3.64G   0.03594   0.01414         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.817      0.857      0.853      0.462
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        91/199     3.64G   0.02979   0.01244         0         9       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.782      0.856      0.855       0.48
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        92/199     3.64G   0.03682    0.0135         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.8      0.952      0.894       0.52
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        93/199     3.64G   0.03082   0.01277         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.769      0.951      0.908      0.408
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        94/199     3.64G   0.03124   0.01398         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.783      0.857      0.866      0.422
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        95/199     3.64G   0.02854   0.01321         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.74      0.949      0.859      0.472
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        96/199     3.64G   0.03051   0.01323         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.792      0.905      0.852      0.504
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        97/199     3.64G   0.03418   0.01315         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.791      0.899      0.878      0.452
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        98/199     3.64G   0.03541   0.01336         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.818      0.857      0.869      0.495
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
        99/199     3.64G   0.03023   0.01238         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.818      0.856      0.866      0.397
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       100/199     3.64G   0.02873   0.01151         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.535      0.714       0.64      0.347
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       101/199     3.64G   0.03632   0.01301         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.515      0.758      0.603      0.336
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       102/199     3.64G   0.03484    0.0123         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.782      0.856       0.84      0.482
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       103/199     3.64G   0.03097   0.01503         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.841      0.762      0.869      0.485
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       104/199     3.64G   0.03694   0.01268         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.809      0.809      0.903      0.379
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       105/199     3.64G   0.04149   0.01437         0        22       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.857      0.857      0.922      0.506
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       106/199     3.64G   0.03128   0.01348         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.856      0.857      0.903      0.511
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       107/199     3.64G   0.02807   0.01284         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.75      0.856      0.881       0.51
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       108/199     3.64G   0.02511   0.01303         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.8      0.761      0.879       0.49
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       109/199     3.64G   0.02304   0.01256         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.8      0.762      0.866      0.472
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       110/199     3.64G   0.03411   0.01478         0        20       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.692      0.857      0.829       0.38
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       111/199     3.64G   0.02543   0.01201         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.769      0.952      0.899       0.48
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       112/199     3.64G   0.03164   0.01306         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.869      0.952       0.93       0.39
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       113/199     3.64G   0.03381   0.01274         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.869      0.952      0.937       0.53
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       114/199     3.64G   0.02691   0.01388         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.87      0.952      0.951      0.544
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       115/199     3.64G   0.02726   0.01248         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.777          1      0.946      0.524
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       116/199     3.64G   0.02616   0.01309         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.749          1      0.937      0.538
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       117/199     3.64G    0.0278   0.01319         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.9      0.857       0.95      0.519
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       118/199     3.64G   0.02744   0.01237         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.857      0.856      0.936      0.507
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       119/199     3.64G   0.02754   0.01289         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.791      0.905      0.919      0.484
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       120/199     3.64G   0.02657   0.01135         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.792      0.904      0.901      0.483
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       121/199     3.64G   0.03267   0.01259         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.864      0.904      0.921      0.491
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       122/199     3.64G    0.0283   0.01195         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.904      0.939      0.563
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       123/199     3.64G   0.02916   0.01306         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.863      0.905      0.927      0.522
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       124/199     3.64G    0.0297   0.01403         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.9      0.857      0.927      0.516
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       125/199     3.64G    0.0261   0.01279         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.826      0.904      0.911      0.518
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       126/199     3.64G   0.02596   0.01286         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.857      0.857      0.916      0.498
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       127/199     3.64G   0.02894   0.01303         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.826      0.905      0.926      0.527
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       128/199     3.64G   0.02405   0.01086         0        10       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.857      0.857      0.911      0.496
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       129/199     3.64G   0.02972   0.01261         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.9      0.857      0.919       0.51
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       130/199     3.64G   0.02488    0.0123         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21        0.9      0.857      0.939       0.51
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       131/199     3.64G   0.02852   0.01246         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.947      0.857      0.949      0.553
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       132/199     3.64G   0.02658   0.01076         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.864      0.905      0.954      0.551
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       133/199     3.64G    0.0193   0.01017         0        10       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.87      0.952      0.957      0.555
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       134/199     3.64G   0.02784   0.01094         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.905      0.972      0.582
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       135/199     3.64G   0.02479   0.01046         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.977      0.566
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       136/199     3.64G   0.02458   0.01049         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.972      0.537
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       137/199     3.64G   0.02284  0.009208         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.905      0.976      0.538
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       138/199     3.64G   0.03005  0.009823         0         8       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.978      0.574
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       139/199     3.64G   0.02549   0.01014         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.982      0.573
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       140/199     3.64G   0.02128   0.01069         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.982      0.586
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       141/199     3.64G   0.02569   0.01146         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.909      0.952      0.974      0.592
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       142/199     3.64G   0.01807   0.01049         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.909      0.952      0.972      0.562
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       143/199     3.64G   0.03048   0.01083         0         8       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.87      0.952      0.971       0.58
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       144/199     3.64G   0.02443   0.01067         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.869      0.952       0.97       0.58
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       145/199     3.64G   0.02486   0.01107         0        21       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.87      0.952      0.973      0.621
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       146/199     3.64G   0.02428   0.01135         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.805          1      0.965        0.6
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       147/199     3.64G   0.02234   0.01069         0        11       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.87      0.952      0.964      0.608
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       148/199     3.64G   0.02011   0.01039         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.869      0.952      0.965      0.614
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       149/199     3.64G   0.02204   0.01039         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.869      0.952      0.965      0.606
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       150/199     3.64G   0.02033   0.01029         0        21       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.87      0.952      0.954      0.618
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       151/199     3.64G   0.02168  0.009698         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.833      0.952      0.959      0.609
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       152/199     3.64G   0.02658   0.01166         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.833      0.952      0.946      0.584
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       153/199     3.64G   0.02024   0.01195         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.833      0.952       0.95      0.606
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       154/199     3.64G   0.02367   0.01171         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.833      0.952       0.95      0.608
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       155/199     3.64G   0.01869    0.0102         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.833      0.952      0.949      0.615
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       156/199     3.64G   0.02102   0.01088         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.905      0.905      0.952      0.632
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       157/199     3.64G   0.02076   0.01056         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.909      0.952      0.957      0.608
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       158/199     3.64G   0.02199   0.01124         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.905      0.959      0.598
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       159/199     3.64G   0.02339   0.01021         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.904      0.958      0.589
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       160/199     3.64G   0.02324   0.01112         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.856      0.955      0.556
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       161/199     3.64G   0.02595    0.0103         0         9       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.904      0.959      0.578
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       162/199     3.64G   0.02022  0.009518         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.979      0.592
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       163/199     3.64G   0.02145   0.01075         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.978      0.592
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       164/199     3.64G   0.01955  0.009416         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.978      0.565
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       165/199     3.64G   0.02051  0.009249         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.979      0.583
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       166/199     3.64G   0.02065  0.009715         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.949      0.905      0.975      0.579
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       167/199     3.64G   0.01916   0.01069         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.905      0.978      0.607
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       168/199     3.64G   0.01948  0.009527         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.979      0.606
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       169/199     3.64G   0.01746  0.008889         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.976       0.61
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       170/199     3.64G   0.02253  0.009758         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.968      0.595
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       171/199     3.64G   0.01564   0.01081         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.958      0.575
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       172/199     3.64G   0.02021  0.008912         0        16       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.958      0.576
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       173/199     3.64G   0.01763  0.009743         0        13       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.955      0.588
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       174/199     3.64G   0.01541  0.008998         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.959      0.587
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       175/199     3.64G   0.01715  0.009817         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.999      0.905      0.957      0.582
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       176/199     3.64G   0.02278    0.0102         0        15       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.949      0.905      0.967      0.577
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       177/199     3.64G   0.01701  0.008705         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.971      0.583
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       178/199     3.64G   0.01945  0.008851         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.971      0.577
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       179/199     3.64G   0.01759  0.007602         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.969      0.561
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       180/199     3.64G   0.01779  0.008252         0         9       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.961      0.556
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       181/199     3.64G   0.02401   0.01074         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.959      0.568
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       182/199     3.64G   0.02392   0.00988         0        22       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.857      0.961      0.544
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       183/199     3.64G   0.02263  0.009115         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.905      0.905      0.959      0.546
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       184/199     3.64G   0.01704   0.00996         0        21       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.905      0.962      0.559
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       185/199     3.64G   0.02246  0.009778         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21       0.95      0.902      0.962       0.55
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       186/199     3.64G   0.01899  0.009308         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.904      0.977      0.555
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       187/199     3.64G   0.01886  0.008656         0        10       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.978      0.556
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       188/199     3.64G   0.02092   0.00885         0        14       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.95      0.552
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       189/199     3.64G   0.01944  0.009833         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.95      0.551
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       190/199     3.64G   0.01872  0.008706         0        19       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.952      0.952       0.98      0.575
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       191/199     3.64G   0.01753  0.009133         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21      0.952      0.952      0.983      0.581
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       192/199     3.64G   0.02013   0.00927         0        18       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905      0.979      0.556
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       193/199     3.64G   0.01749  0.009514         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.98      0.566
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       194/199     3.64G   0.01585  0.009666         0        12       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.98      0.569
    
         Epoch   gpu_mem       box       obj       cls    labels  img_size
       195/199     3.64G   0.02206  0.009072         0        17       640: 100%|███
                   Class     Images     Labels          P          R     mAP@.5 mAP@
                     all         24         21          1      0.905       0.98       0.57
    
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
    Results saved to [1mruns/train/hairTets11[0m



```python
cd yolov5
```

    /workspace/Docker/yolo_skin/yolov5



```python
!python detect.py --weight /workspace/Docker/yolo_skin/yolov5/runs/train/hairTets11/weights/last.pt --img 640 --conf 0.7 --source /workspace/Docker/yolo_skin/images
```

    [34m[1mdetect: [0mweights=['/workspace/Docker/yolo_skin/yolov5/runs/train/hairTets11/weights/last.pt'], source=/workspace/Docker/yolo_skin/images, imgsz=[640, 640], conf_thres=0.7, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False
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
    Results saved to [1mruns/detect/exp11[0m

