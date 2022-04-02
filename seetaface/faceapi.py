# -*- encoding: utf-8 -*-
'''
Filename      : faceapi.py
Description   : seetaface6python
Author  	  : Yang Jian
Contact 	  : lian01110@outlook.com
Time          : 2021/11/03 17:15:49
IDE           : PYTHON
REFERENCE 	  : https://github.com/yangjian1218
'''

import base64
import os
import platform

import cv2
import numpy as np

from .face_struct import *

API_DIR = os.path.dirname(os.path.abspath(__file__))  # api.py的目录文件夹的绝对路径
# print("API_DIR:", API_DIR)  # /home/yangjian/Projects/FaceAPI_demo
platform_name = platform.platform().lower()
# print("-----------------------platform_name:", platform_name)
LIB_PATH = None
dll = None
# if "windows" in platform_name:
#     LIB_PATH = os.path.join(API_DIR, "lib", "win")
#     os.environ["PATH"] += os.pathsep + LIB_PATH
#     dll = CDLL(os.path.join(LIB_PATH, "libFaceAPI.dll"))
# elif "ubuntu" in platform_name or "debian" in platform_name:
#     print("ubuntu")
#     LIB_PATH = os.path.join(API_DIR, "lib", "ubuntu")
#     print("lib_path", LIB_PATH)
#     dll = CDLL(os.path.join(LIB_PATH, "libFaceAPI.so"))
# else:
#     # todo centos字样不会在platform.platform()中显示..例如输出为:'linux-3.10.0-1160.25.1.el7.x86_64-x86_64-with-glibc2.10'
#     # /home/yangjian/Projects/FaceAPI_demo/lib/centos
#     LIB_PATH = os.path.join(API_DIR, "lib", "centos")
#     dll = CDLL(os.path.join(LIB_PATH, "libSeetaFaceAPI.so"))

# todo  只能在linux下使用
LIB_PATH = os.path.join(API_DIR, "lib")
dll = CDLL(os.path.join(LIB_PATH, "libSeetaFaceAPI.so"))

# /home/yangjian/Projects/FaceAPI_demo/model
MODEL_DIR = os.path.join(API_DIR, "model")

# 由于传递int数组比较方便,所以把字符串映射给int
func_dict = {"FACE_DETECT": 0, "LANDMARKER5": 1, "LANDMARKER68": 2, "LIVENESS": 3, "LANDMARKER_MASK": 4, "FACE_AGE": 5,
             "FACE_GENDER": 6, "MOUTH_MASK": 7, "EYE_STATE": 8, "FACE_CLARITY": 9, "FACE_BRIGHT": 10, "FACE_RESOLUTION": 11,
             "FACE_POSE": 12, "FACE_INTEGRITY": 13, "FACE_TRACK": 14, "FACE_RECOGNITION": 15}


# 人脸检测器属性设置枚举
class DetectProperty():
    # 最小人脸 默认值大小为20
    PROPERTY_MIN_FACE_SIZE = 0
    # 默认为0.9
    PROPERTY_THRESHOLD = 1  # 检测器阈值
    PROPERTY_MAX_IMAGE_WIDTH = 2  # 可检测的图像最大宽度
    PROPERTY_MAX_IMAGE_HEIGHT = 3  # 可检测的图像最大高度
    # 默认为1
    PROPERTY_NUMBER_THREADS = 4  # 可检测的图像人脸最大数量


def get_numpy_by_seetaImageData(image_data: SeetaImageData) -> np.array:
    """

    param  image_data:SeetaImageData结构体
    return  :numpy数组
    """

    width = image_data.width
    height = image_data.height
    channels = image_data.channels
    row_array = np.array(np.fromiter(
        image_data.data, dtype=np.uint8, count=width * height * channels))
    image_np = row_array.reshape([height, width, channels])
    return image_np


def get_seetaImageData_by_numpy(image_np: np.array) -> SeetaImageData:
    """

    param  image_np:numpy数组
    return  :seetaImageData结构体
    """

    seetaImageData = SeetaImageData()
    height, width, channels = image_np.shape
    seetaImageData.height = int(height)
    seetaImageData.width = int(width)
    seetaImageData.channels = int(channels)
    seetaImageData.data = image_np.ctypes.data_as(POINTER(c_ubyte))
    return seetaImageData


def get_numpy_by_cvImage(cvimage):
    """
    结构体转为numpy图片
    param  cvimage:cvimage的结构体,包含rows,cols,channels,data
    return  :numpy图片
    """

    data = cvimage.data
    cv_rows = cvimage.rows
    cv_cols = cvimage.cols
    cv_channels = cvimage.channels
    b = string_at(data, cv_cols * cv_rows * cv_channels)  # 类似于base64
    nparr = np.frombuffer(b, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_decode


def get_seetaRect_by_list(box):
    """
    从retinaface获取的box转为seetaRect
    param  box:retinaface获取的box
    return  :seetaRect
    """

    seetaRect = SeetaRect()
    seetaRect.x = int(box[0])
    seetaRect.y = int(box[1])
    seetaRect.width = int(box[2] - box[0])
    seetaRect.height = int(box[3] - box[1])
    return seetaRect


def get_seetaPointF_by_point(point):
    seetaPointF = SeetaPointF()
    seetaPointF.x = point[0]
    seetaPointF.y = point[1]
    return seetaPointF


def get_seetaPointFList_by_point5(points5):
    seetaPointFList = (SeetaPointF * 5)()
    for i in range(5):
        seetaPointFList[i] = get_seetaPointF_by_point(points5[i])
    return seetaPointFList


class SeetaFace(object):
    def __init__(self, funcs_list, device=0, id=0):
        """

        param  funcs_list:功能列表
        param  device:处理器, 自动:0, cpu:1,gpu:2
        return  :
        """

        self._funcs_list = funcs_list
        self.funcs_len = len(funcs_list)
        self._dll_func_def()
        # self._init_engine(model_path)
        self._set_device(device, id)

    def check_init(self, init_flag):
        if not (init_flag in self._funcs_list):
            # init_flag 跟_funcs_list 的交集 是否为空
            raise Exception("该功能对应的引擎未初始化!")

    def _dll_func_def(self):
        # 类方法与dll方法对应,以及规定入参与返回的类型
        self._Track = dll.Track  # 人脸跟踪
        self._Track.restype = SeetaTrackingFaceInfoArray
        self._Track.argtypes = (POINTER(SeetaImageData),)

        # todo 新增人脸跟踪视频分辨率设置,不然针对不同视频,没办法适应
        self._SetTrackResolution = dll.SetTrackResolution
        self._SetTrackResolution.argtypes = (c_int32, c_int32)

        # 这里与原作者不同,原作者应该写错了.
        self._SetSingleCalculationThreads = dll.SetSingleCalculationThreads
        self._SetSingleCalculationThreads.argtypes = (c_int32,)

        self._SetInterval = dll.SetInterval  # 人脸跟踪间隔
        self._SetInterval.argtypes = (c_int32,)

        self._SetMinFaceSize = dll.SetMinFaceSize  # 设置人脸跟踪最小尺寸
        self._SetMinFaceSize.argtypes = (c_int32,)

        self._SetThreshold = dll.SetThreshold  # 设置人脸跟踪人脸置信度阈值
        self._SetThreshold.argtypes = (c_float,)

        self._Reset = dll.Reset  # 人脸跟踪清除

        self._Predict = dll.Predict  # 活体检测
        self._Predict.restype = c_int32
        self._Predict.argtypes = (POINTER(SeetaImageData), POINTER(
            SeetaRect), POINTER(SeetaPointF))

        # 活体检测清晰度与真实值阈值(clarity,reality)
        self._SetLiveThreshold = dll.SetLiveThreshold
        self._SetLiveThreshold.argtypes = (c_float, c_float)

        self._PredictVideo = dll.PredictVideo
        self._PredictVideo.restype = c_int32
        self._PredictVideo.argtypes = (
            POINTER(SeetaImageData), POINTER(SeetaRect), POINTER(SeetaPointF))

        self._ResetVideo = dll.ResetVideo

        self._GetPreFrameScore = dll.GetPreFrameScore
        self._GetPreFrameScore.argtypes = (POINTER(c_float), POINTER(c_float))

        self._mark5 = dll.mark5
        self._mark5.restype = c_int32
        self._mark5.argtypes = (POINTER(SeetaImageData), POINTER(
            SeetaRect), POINTER(SeetaPointF))

        self._mark68 = dll.mark68
        self._mark68.restype = c_int32
        self._mark68.argtypes = (POINTER(SeetaImageData), POINTER(
            SeetaRect), POINTER(SeetaPointF))

        self._markMask = dll.markMask  # 人脸遮挡检测
        self._markMask.restype = c_int32
        self._markMask.argtypes = (POINTER(SeetaImageData), POINTER(
            SeetaRect), POINTER(SeetaPointF), POINTER(c_int32))

        self._CropFace = dll.CropFace
        self._CropFace.restype = SeetaImageData
        self._CropFace.argtypes = (
            POINTER(SeetaImageData), POINTER(SeetaPointF))

        self._ExtractCroppedFace = dll.ExtractCroppedFace
        self._ExtractCroppedFace.restype = c_int32
        self._ExtractCroppedFace.argtypes = (
            POINTER(SeetaImageData), POINTER(c_float))

        self._Extract = dll.Extract
        self._Extract.restype = c_int32
        self._Extract.argtypes = (
            POINTER(SeetaImageData), POINTER(SeetaPointF), POINTER(c_float))

        self._CalculateSimilarity = dll.CalculateSimilarity
        self._CalculateSimilarity.restype = c_float
        self._CalculateSimilarity.argtypes = (
            POINTER(c_float), POINTER(c_float))

        self._Detect = dll.Detect  # 人脸检测
        self._Detect.restype = SeetaFaceInfoArray
        self._Detect.argtypes = (POINTER(SeetaImageData),)

        self._SetProperty = dll.SetProperty  # 设置人脸属性值阈值,如最小尺寸(宽) 跟置信度
        self._SetProperty.argtypes = (c_int32, c_double)

        # todo 取消人脸姿态角的深度方法检测,改为传统方法
        # self._check = dll.check   #检测人脸姿态角度质量检测
        # self._check.restype = c_int32
        # self._check.argtypes = (POINTER(SeetaImageData), POINTER(
        #     SeetaRect), POINTER(SeetaPointF))

        # self._set = dll.set   # 设置人脸姿态评估(深度)的参数
        # self._set.argtypes = (c_int32, c_int32, c_int32,
        #                       c_int32, c_int32, c_int32)

        self._PredictGenderWithCrop = dll.PredictGenderWithCrop
        self._PredictGenderWithCrop.restype = c_int32
        self._PredictGenderWithCrop.argtypes = (
            POINTER(SeetaImageData), POINTER(SeetaPointF))

        self._PredictGender = dll.PredictGender
        self._PredictGender.restype = c_int32
        self._PredictGender.argtypes = (POINTER(SeetaImageData),)

        self._PredictAgeWithCrop = dll.PredictAgeWithCrop
        self._PredictAgeWithCrop.restype = c_int32
        self._PredictAgeWithCrop.argtypes = (
            POINTER(SeetaImageData), POINTER(SeetaPointF))

        self._PredictAge = dll.PredictAge
        self._PredictAge.restype = c_int32
        self._PredictAge.argtypes = (POINTER(SeetaImageData),)

        self.InitEngine = dll.InitEngine
        self.InitEngine.restype = c_int32
        self.InitEngine.argtypes = (c_int32 * self.funcs_len, c_int32)

        # todo 新增部分
        self.get_modelpath = dll.get_modelpath  # 获取模型文件目录,这样可以把模型文件位置乱放
        self.get_modelpath.argtypes = [c_char_p]
        self.get_modelpath.restype = c_void_p

        self.set_device = dll.set_device  # 设置是否使用cpu/gpu 以及gpu的标号
        self.set_device.argtypes = [c_int32, c_int32]

        self._DetectMask = dll.DetectMask  # 口罩检测
        self._DetectMask.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaRect)]
        self._DetectMask.restype = c_int32

        self._DectectEye = dll.DectectEye  # 眼睛状态检测,睁闭眼
        self._DectectEye.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaPointF), c_int32 * 2]
        self._DetectMask.restype = c_void_p

        self._ClarityEvaluate = dll.ClarityEvaluate  # 清晰度评估(传统)
        self._ClarityEvaluate.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaRect), POINTER(SeetaPointF)]
        self._ClarityEvaluate.restype = c_char_p

        self._BrightEvaluate = dll.BrightEvaluate  # 亮度度评估(传统)
        self._BrightEvaluate.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaRect), POINTER(SeetaPointF)]
        self._BrightEvaluate.restype = c_char_p

        self._ResolutionEvaluate = dll.ResolutionEvaluate  # 分辨率评估(传统)
        self._ResolutionEvaluate.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaRect), POINTER(SeetaPointF)]
        self._ResolutionEvaluate.restype = c_char_p

        self._PoseEvaluate = dll.PoseEvaluate  # 人脸姿态质量(传统)
        self._PoseEvaluate.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaRect), POINTER(SeetaPointF)]
        self._PoseEvaluate.restype = c_char_p

        self._IntegrityEvaluate = dll.IntegrityEvaluate  # 人脸完整性评估(传统)
        self._IntegrityEvaluate.argtypes = [
            POINTER(SeetaImageData), POINTER(SeetaRect), POINTER(SeetaPointF)]
        self._IntegrityEvaluate.restype = c_char_p

    def _set_device(self, device, id):
        self.set_device(c_int32(device), c_int32(id))

    def init_engine(self, modelpath):
        modelpath = os.path.abspath(modelpath)
        modelpath = c_char_p(modelpath.encode())
        self.get_modelpath(modelpath)
        cwd = os.getcwd()  # 获取当前工作目录, 并不是该文件的目录
        # print("目录地址上一级:",os.path.dirname(os.path.abspath(__file__)) + "/../")
        # 获取该文件的绝对路径- 返回目录地址,--返回目录上一级 --切换工作目录为 ;d:\AI\Face\seetaFace6Python
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")
        Array = c_int * self.funcs_len
        func_list = Array()
        for i in range(self.funcs_len):
            func_list[i] = func_dict[self._funcs_list[i]]
        self.InitEngine(func_list, self.funcs_len)  # 初始化模型
        os.chdir(cwd)  # 切换目录会当前工作目录

    def Track(self, simage) -> SeetaTrackingFaceInfoArray:
        """
        检测图像中的位置信息，
        追踪模式下和检测模式下返回的检测结果相似
        但是追踪模式下会额外多 人脸追踪id（PID），frame_no，step等一些额外参数
        大部分情况下只用关心其中的PID参数(为每一个出现的人脸分配的id，从0开始)
        :param simage: 原图的seetaImageData
        :return:
        """
        self.check_init("FACE_TRACK")

        return self._Track(simage)

    def SetTrackResolution(self, width, height):
        """
        设置人脸跟踪视频的分辨率
        param  width:分辨率的宽
        param  height:分辨率的高
        return  :
        """
        self._SetTrackResolution(width, height)

    def SetSingleCalculationThreads(self, thread_num: int):
        """
        设置追踪处理的线程数
        :param thread_num:
        :return:
        """
        self._SetSingleCalculationThreads(thread_num)

    def SetInterval(self, interval: int):
        """
        设置追踪过程中的检测间隔
        间隔默认值为10。这里跟踪间隔是为了发现新增PID的间隔。
        检测器会通过整张图像检测人脸去发现是否有新增的PID，
        所以这个值太小会导致跟踪速度变慢（不断做全局检测）
        这个值太大会导致画面中新增加的人脸不会立马被跟踪到
        :param interval: 检测间隔帧数
        :return: None
        """
        self._SetInterval(interval)

    def SetMinFaceSize(self, size: int):
        """
        设置人脸追踪最小检测人脸大小，默认已设置20
        :param size:
        :return:
        """
        self._SetMinFaceSize(size)

    def SetThreshold(self, clarity: float):
        # 人脸跟踪人脸置信度
        self._SetThreshold(clarity)

    def Reset(self):
        """
        人脸跟踪模块 重置，更换视频源 时需要调用
        :return:
        """
        self._Reset()

    def Detect(self, simage) -> SeetaFaceInfoArray:
        """
        人脸检测
        :param frame: 原始图像
        :return: 人脸检测信息数组
        """

        self.check_init("FACE_DETECT")  # 检验模型是否在输入模型参数中,这里并没有初始化
        return self._Detect(simage)

    def SetProperty(self, property: int, value):
        """
        人脸检测属性设置
        param  property:属性名称. 最小人脸尺寸1:PROPERTY_MIN_FACE_SIZE,默认20 置信度2:PROPERTY_THRESHOLD 
        图像最大宽度3:PROPERTY_MAX_IMAGE_WIDTH  最大高度4PROPERTY_MAX_IMAGE_HEIGHT
        return  :
        """
        self._SetProperty(property, value)

    def Predict(self, simage: SeetaImageData, face: SeetaRect,
                points: List[SeetaPointF]) -> int:
        """
        单帧rgb活体检测
        :param simage: 原图的SeetaImageData
        :param face: 人脸区域
        :param points:  人脸关键点位置
        :return:  活体检测结果
        0:真实人脸
        1:攻击人脸（假人脸）
        2:无法判断（人脸成像质量不好）
        """
        self.check_init("LIVENESS")
        islive = self._Predict(simage, face, points)
        return islive

    def SetLiveThreshold(self, clarity, reality):
        # 设置活体检测的清晰度与真实度的阈值
        self._SetLiveThreshold(clarity, reality)

    def PredictVideo(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]) -> int:
        """
        视频rgb活体检测（多帧判断出的结果）
        相比较于Predict 函数，多了一个正在检测状态的返回值
        :param simage: 原图的SeetaImageData
        :param face: 人脸区域
        :param points:  人脸关键点位置
        :return:  活体检测结果
        0:真实人脸
        1:攻击人脸（假人脸）
        2:无法判断（人脸成像质量不好）
        3: 正在检测
        """
        self.check_init("LIVENESS")
        return self._PredictVideo(simage, face, points)

    def ResetVideo(self):
        self._ResetVideo()

    def GetPreFrameScore(self):
        clarity = c_float()
        reality = c_float()
        self._GetPreFrameScore(clarity, reality)
        return (clarity.value, reality.value)

    def mark5(self, simage: SeetaImageData, face: SeetaRect) -> List[SeetaPointF]:
        """
        给定一张原始图片，和其中人脸区域，返回该人脸区域中5个关键点位置 [左眼，右眼，鼻子，左边嘴角，右边嘴角]
        :param simage: 原图的SeetaImageData
        :param face: 人脸区域位置
        :return:
        """
        self.check_init("LANDMARKER5")
        points = (SeetaPointF * 5)()
        self._mark5(simage, face, points)
        return points

    def mark68(self, simage: SeetaImageData, face: SeetaRect) -> List[SeetaPointF]:
        """
        给定一张原始图片，和其中人脸区域，返回该人脸区域中的68个关键点位置
        :param simage: 原图的SeetaImageData
        :param face: 人脸区域位置
        :return:
        """
        self.check_init("LANDMARKER68")
        points = (SeetaPointF * 68)()
        self._mark68(simage, face, points)
        return points

    def markMask(self, simage: SeetaImageData, face: SeetaRect) -> (List[SeetaPointF], int):
        """
        给定一张原始图片，和其中人脸区域，返回该人脸区域中的5个关键点位置,
        和这 5点是否被遮挡的数组 [左眼，右眼，鼻子，左边嘴角，右边嘴角]
        :param simage: 原图的SeetaImageData
        :param face: 人脸区域位置
        :return:
        points：5关键点数组
       face_mask： 如戴口罩时 基本会返回数组【0，0，1，1，1】，0：没被遮挡  1：被遮挡
        """
        self.check_init("LANDMARKER_MASK")
        points = (SeetaPointF * 5)()
        face_mask = (c_int32 * 5)()
        self._markMask(simage, face, points, face_mask)
        face_mask_list = []  # 由于数组没办法直接打印,所以转换列表
        for i in range(5):
            face_mask_list.append(face_mask[i])
        return points, face_mask_list

    def CropFace(self, simage: SeetaImageData, points: List[SeetaPointF]):
        """
        根据关键点位置，裁剪出矫正后的人脸区域图片
        :param simage: 原图的SeetaImageData
        :param points:
        :return: [256*256*c]
        """
        out_seetaImageData = self._CropFace(simage, points)
        return get_numpy_by_seetaImageData(out_seetaImageData)

    def ExtractCroppedFace(self, frame: np.array):
        """
        #提取人脸图像特征值（整个一张图片为一张人脸时）
        :param frame:
        :return:
        """
        self.check_init("FACE_RECOGNITION")
        if frame.shape[0] != 256 or frame.shape[1] != 256:
            seetaImageData = get_seetaImageData_by_numpy(
                cv2.resize(frame, (256, 256)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        feature = (c_float * 1024)()
        self._ExtractCroppedFace(seetaImageData, feature)
        return feature

    def Extract(self, simage: SeetaImageData, points: List[SeetaPointF]):
        """
        在一张图片中提取指定人脸关键点区域的人脸的特征值
        :param simage: 原图的SeetaImageData
        :param points:
        :return:
        """
        self.check_init("FACE_RECOGNITION")
        feature = (c_float * 1024)()
        self._Extract(simage, points, feature)
        return feature

    def CalculateSimilarity(self, features1, features2):
        self.check_init("FACE_RECOGNITION")
        return self._CalculateSimilarity(features1, features2)

    def compare_feature_np(self, feature1: np.array, feature2: np.array) -> float:
        """
        使用numpy 计算，比较人脸特征值相似度
       :param feature1: 人脸特征值1
        :param feature2: 人脸特征值2
        :return: 人脸相似度
        """
        dot = np.sum(np.multiply(feature1, feature2))
        norm = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        dist = dot / norm
        return float(dist)

    def get_feature_by_byte(self, feature_byte: bytes):
        """
        通过特征值二进制 获取 feature 数据
        :param feature:
        :return:
        """
        feature = np.frombuffer(feature_byte).ctypes.data_as(POINTER(c_float))
        return feature

    def get_feature_byte(self, feature: List[c_float]):
        """
        获取 feature 的字节流表示数据
        :param feature:
        :return:
        """
        return string_at(feature, 1024 * 4)

    def get_feature_numpy(self, feature: List[c_float]) -> np.array:
        """
        获取 feature 的numpy表示数据
        :param feature:
        :return:
        """
        face_encoding = (
            np.frombuffer(string_at(feature, 1024 * 4), dtype=np.float32))
        return face_encoding

    def get_feature_base64(self, feature: List[c_float]):
        """
        获取 feature 的base64表示形式
        :param feature:
        :return: base64 字符串
        """
        return base64.b64encode(self.get_feature_byte(feature)).decode(encoding="UTF-8")

    def check(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]) -> int:
        """
        #检测人脸姿态角度是否合适
        :param simage: 原图的SeetaImageData
        :param face:
        :param points:
        :return:  0：低  1：中等 2：高
        """
        self.check_init("FACE_POSE_EX")
        return self._check(simage, face, points)

    def set(self, yaw_low_threshold: int,
            yaw_high_threshold: int,
            pitch_low_threshold: int,
            pitch_high_threshold: int,
            roll_low_threshold: int,
            roll_high_threshold: int):
        """
        设置人脸姿态角度评估模型判定范围
        :param yaw_low_threshold:
        :param yaw_high_threshold:
        :param pitch_low_threshold:
        :param pitch_high_threshold:
        :param roll_low_threshold:
        :param roll_high_threshold:
        """
        self._set(yaw_low_threshold, yaw_high_threshold, pitch_low_threshold,
                  pitch_high_threshold, roll_low_threshold, roll_high_threshold)

    def PredictGenderWithCrop(self, simage: SeetaImageData, points: List[SeetaPointF]) -> int:
        """
        检测一张原图中一个人脸的性别，需要人脸关键点位置
        :param simage: 原图的SeetaImageData
        :param points: 人脸关键点
        :return: 0：男   1：女
        """
        self.check_init("FACE_GENDER")
        return self._PredictGenderWithCrop(simage, points)

    def PredictGender(self, frame: np.array) -> int:
        """
        检测一张只有人脸的图片,识别出性别
        :param simage: 原图
        :return: 0：男   1：女
        """
        self.check_init("FACE_GENDER")
        if frame.shape[0] != 128 or frame.shape[1] != 128:
            seetaImageData = get_seetaImageData_by_numpy(
                cv2.resize(frame, (128, 128)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictGender(seetaImageData)

    def PredictAgeWithCrop(self, simage: SeetaImageData, points: List[SeetaPointF]) -> int:
        """
        检测一张原图中一个人脸的年龄，需要人脸关键点位置
        :param simage: 原图的SeetaImageData
        :param points: 人脸关键点
        :return: 年龄大小
        """
        self.check_init("FACE_AGE")
        return self._PredictAgeWithCrop(simage, points)

    def PredictAge(self, frame: np.array) -> int:
        """
        检测一张只有人脸的图片,识别出年龄
        :param frame: 原图
        :return: 年龄大小
        """
        self.check_init("FACE_AGE")
        if frame.shape[0] != 256 or frame.shape[1] != 256:
            seetaImageData = get_seetaImageData_by_numpy(
                cv2.resize(frame, (256, 256)))
        else:
            seetaImageData = get_seetaImageData_by_numpy(frame)
        return self._PredictAge(seetaImageData)

    def DetectMask(self, simage: SeetaImageData, face: SeetaRect) -> int:
        """
        口罩检测
        :param simage: 原图的SeetaImageData
        :param face : 人脸框
        return  :是否戴了口罩, 0没戴, 1戴了
        """
        self.check_init("MOUTH_MASK")
        return self._DetectMask(simage, face)

    def DetectEye(self, simage: SeetaImageData, points: List[SeetaPointF]):
        """
        人眼状态检测(传统)
        :param simage: 原图的SeetaImageData
        :param points: 人脸关键点
        return  :
        """
        statedict = {0: "close", 1: "open", 2: "random", 3: "unknown"}
        self.check_init("EYE_STATE")
        eyestate = (c_int32 * 2)()
        self._DectectEye(simage, points, eyestate)
        state = {
            "left": statedict[eyestate[0]],
            "right": statedict[eyestate[1]],
        }
        return state

    def ClarityEvaluate(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]):
        """
        清晰度评估(传统)
        :param simage: 原图的SeetaImageData
        :param face : 人脸框
        :param points: 人脸关键点
        return  :"LOW", "MEDIUM", "HIGH" 
        """
        self.check_init("FACE_CLARITY")
        level = self._ClarityEvaluate(simage, face, points)
        return level.decode()

    def BrightEvaluate(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]):
        """
        明亮度评估(传统)
        :param simage: 原图的SeetaImageData
        :param face : 人脸框
        :param points: 人脸关键点
        return  :"LOW", "MEDIUM", "HIGH" 
        """
        self.check_init("FACE_BRIGHT")
        level = self._BrightEvaluate(simage, face, points)
        return level.decode()

    def ResolutionEvaluate(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]):
        """
        分辨率评估(传统)
        :param simage: 原图的SeetaImageData
        :param face : 人脸框
        :param points: 人脸关键点
        return  :"LOW", "MEDIUM", "HIGH" 
        """
        self.check_init("FACE_RESOLUTION")

        level = self._BrightEvaluate(simage, face, points)
        return level.decode()

    def PoseEvaluate(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]):
        """
        人脸姿态质量评估
        :param simage: 原图的SeetaImageData
        :param face : 人脸框
        :param points: 人脸关键点
        return  :"LOW", "MEDIUM", "HIGH" 
        """
        self.check_init("FACE_POSE")

        level = self._PoseEvaluate(simage, face, points)
        return level.decode()

    def IntegrityEvaluate(self, simage: SeetaImageData, face: SeetaRect, points: List[SeetaPointF]):
        """
        人脸完整性评估
        :param simage: 原图的SeetaImageData
        :param face : 人脸框
        :param points: 人脸关键点
        return  :"LOW", "MEDIUM", "HIGH" 
        """
        self.check_init("FACE_INTEGRITY")

        level = self._IntegrityEvaluate(simage, face, points)
        return level.decode()
