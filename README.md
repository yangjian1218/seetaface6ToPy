# seetaface6ToPy

## 1.简介

项目基于`SeetaFace6` 官方地址为:https://github.com/SeetaFace6Open/index

灵感来源:https://github.com/tensorflower/seetaFace6Python

由于上面的版本没有提供FaceAPI的源码,而且不支持GPU,且功能并没有写全, 故自己利用ctypes编写了一个API.由于我的电脑windows系统编译seetaface老是出错,原因是电脑安装了VS又有MinGW,编译会自动寻找vs,但是又提示在编译某些文件时'cl.exe'找不到,可是我已经设置过环境变量了,百度了很多方案,都不行。(另外这个错误在安装insightface最新版本也出现,如果有知道解决方案的请告诉我)。所以放弃了windows版本,只编译了linux的.linux版本不区分centos跟ubuntu,都可以使用.设置方法一样.

## 2.模型下载

模型可以在seetaface的官网下载:

模型文件：
Part I: [Download](https://pan.baidu.com/s/1LlXe2-YsUxQMe-MLzhQ2Aw) code: `ngne`, including: `age_predictor.csta`, `face_landmarker_pts5.csta`, `fas_first.csta`, `pose_estimation.csta`, `eye_state.csta`, `face_landmarker_pts68.csta`, `fas_second.csta`, `quality_lbn.csta`, `face_detector.csta`, `face_recognizer.csta`, `gender_predictor.csta`, `face_landmarker_mask_pts5.csta`, `face_recognizer_mask.csta`, `mask_detector.csta`.
Part II: [Download](https://pan.baidu.com/s/1xjciq-lkzEBOZsTfVYAT9g) code: `t6j0`，including: `face_recognizer_light.csta`.

## 3.编译

```
cd seetaface
mkdir build && cd build
cmake ..
make
```

将在lib文件夹下产生libSeetaFaceAPI.so文件

## 4. 设置动态库到环境中

```
sudo echo  ${seetaFace6ToPy目录路径}/seetaface/lib/ > /etc/ld.so.conf.d/seetaface6.con  
sudo ldconfig
```

## 5.运行seeta_test.py

使用说明:

1.把需要调用的功能放进一个列表中,名称必须参照下面:

```python
func_list =["FACE_DETECT","LANDMARKER5","LIVENESS","LANDMARKER_MASK","FACE_AGE","FACE_GENDER","FACE_RECOGNITION","MOUTH_MASK","EYE_STATE","FACE_CLARITY","FACE_BRIGHT","FACE_RESOLUTION","FACE_POSE","FACE_INTEGRITY", "FACE_TRACK"]
```

2.设置使用cpu/gpu,并初始化

```
model_path = "./seeta/model"
seetaFace = SeetaFace(func_list,device=0,id=0)
seetaFace.init_engine(model_path)
```

device: ,自动:0, cpu:1, gpu:2 (经过实践:自动并不自动,只会调cpu)

id:处理器的编号,如果为cpu,为0就可以.

## 6.如果觉得有用,点个小星星吧.


