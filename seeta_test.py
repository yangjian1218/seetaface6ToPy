# -*- encoding: utf-8 -*-
'''
Filename      : seeta_test.py
Description   : 
Author  	  : Yang Jian
Contact 	  : lian01110@outlook.com
Time          : 2021/11/03 17:16:17
IDE           : PYTHON
REFERENCE 	  : https://github.com/yangjian1218
'''

from seetaface.faceapi import *
import cv2

func_list = ["FACE_DETECT", "LANDMARKER5", "LIVENESS", "LANDMARKER_MASK", "FACE_AGE", "FACE_GENDER", "FACE_RECOGNITION",
             "MOUTH_MASK", "EYE_STATE", "FACE_CLARITY", "FACE_BRIGHT", "FACE_RESOLUTION", "FACE_POSE", "FACE_INTEGRITY", "FACE_TRACK"]
model_path = "./seetaface/model"
seetaFace = SeetaFace(func_list, device=0, id=0)
seetaFace.SetTrackResolution(310, 310)
seetaFace.init_engine(model_path)

image = cv2.imread("./images/yj.jpg")  # 原图
simage = get_seetaImageData_by_numpy(image)  # 原图转SeetaImageData

# 人脸检测
detect_result = seetaFace.Detect(simage)
rect_list = detect_result.data
print("result", detect_result)

face1 = detect_result.data[0].pos
print("_face:", face1)
# 特征点检测
points5 = seetaFace.mark5(simage, face1)  # 5特征点检测
points68 = seetaFace.mark5(simage, face1)  # 68特征点检测
for i in range(5):
    print("x=", points5[i].x, "	y=", points5[i].y)
# 活体检测
# 0: 真实人脸
# 1: 攻击人脸（假人脸）
# 2: 无法判断（人脸成像质量不好）
livnees = seetaFace.Predict(simage, face1, points5)
clarity, reality = seetaFace.GetPreFrameScore()
print("livnees:", livnees)
print("clarity=", clarity, " reality=", reality)
# 脸部遮挡检测,1为遮挡,0为未遮挡
points_mask, face_mask_list = seetaFace.markMask(simage, face1)
print("face_mask_list:", face_mask_list)
# 年龄检测
age = seetaFace.PredictAgeWithCrop(simage, points5)
print("age:", age)
# 性别检测
gender_ret = seetaFace.PredictGenderWithCrop(simage, points5)
gender = "male" if gender_ret == 0 else "female"
print("gender:", gender)
# 特征提取
feature1 = seetaFace.Extract(simage, points5)
feature_list = []
for i in range(1024):
    feature_list.append(feature1[i])
# print("feature:",feature_list)

image2 = cv2.imread("./images/yj.jpg")  # 原图
simage2 = get_seetaImageData_by_numpy(image2)  # 原图转SeetaImageData
# 计算两个特征值的形似度
detect_result2 = seetaFace.Detect(simage2)
face2 = detect_result2.data[0].pos
points2 = seetaFace.mark5(simage2, face2)
feature2 = seetaFace.Extract(simage2, points2)
similar1 = seetaFace.CalculateSimilarity(feature1, feature2)
print("相似度=", similar1)

# 人脸截取
cropface1 = seetaFace.CropFace(simage, points5)
cv2.imwrite("./images/ch1_crop.jpg", cropface1)
# 人脸跟踪
# track_image = image.copy()
# detectr_result = seetaFace.Track(simage)
#
# face_tr = detectr_result.data[0].pos
# print("face_tr",face_tr)
# PID = detectr_result.data[0].PID
# cv2.rectangle(track_image, (face_tr.x, face_tr.y), (face_tr.x + face_tr.width, face_tr.y + face_tr.height),(255, 0, 0), 2)
# cv2.putText(track_image,"pid:{}".format(PID),(face_tr.x,face_tr.y),1,1,(0,0,255))
# cv2.imshow("track",track_image)
# cv2.waitKey(0)
# cv2.imwrite("./images/track_image.jpg",track_image)
# print("PID=",PID)  # PID为计数
# 口罩检测
mask_ret = seetaFace.DetectMask(simage, face1)
print("mask_ret:", mask_ret)
# 眼睛状态检测
eye_state = seetaFace.DetectEye(simage2, points2)
print("eye_state:", eye_state)
# 清晰度
clarity_level = seetaFace.ClarityEvaluate(simage, face1, points5)
print("清晰度质量:", clarity_level)
# 明亮度
bright_level = seetaFace.BrightEvaluate(simage, face1, points5)
print("明亮度质量:", bright_level)
# 分辨率质量
resolution_level = seetaFace.ClarityEvaluate(simage, face1, points5)
print("分辨率质量:", resolution_level)
# 人脸姿态角质量
pose_level = seetaFace.BrightEvaluate(simage, face1, points5)
print("人脸姿态角质量:", pose_level)
# 人脸完整性质量
integrity_level = seetaFace.BrightEvaluate(simage, face1, points5)
print("人脸完整性质量:", integrity_level)
