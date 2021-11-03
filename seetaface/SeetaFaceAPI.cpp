/*
* @Author: jerry
* @Date:   2021-09-30 11:15:45
* @Last Modified by:   jerry
* @Last Modified time: 2021-09-30 11:18:00
*/
#include <iostream>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <seeta/Common/Struct.h>
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceAntiSpoofing.h>
#include <seeta/MaskDetector.h>        //口罩检测
#include <seeta/EyeStateDetector.h>    //眼睛状态检测
#include <seeta/AgePredictor.h>        //年龄检测
#include <seeta/GenderPredictor.h>     //性别检测
#include <seeta/QualityStructure.h>    //遮挡评估
#include <seeta/QualityOfBrightness.h> //亮度评估
#include <seeta/QualityOfResolution.h> //分辨率评估
#include <seeta/QualityOfIntegrity.h>  //完整性评估
#include <seeta/QualityOfClarity.h>    //清晰度检测(传统)
#include <seeta/QualityOfPose.h>       //姿态评估(传统)
#include <seeta/FaceRecognizer.h>      //人脸识别
#include <seeta/FaceTracker.h>         //人脸跟踪

using namespace std;
// using namespace cv;

seeta::FaceDetector *faceDetector;           //人脸框
seeta::FaceLandmarker *landDetector5;        //5特征点
seeta::FaceLandmarker *landDetector68;       //68特征点
seeta::FaceAntiSpoofing *liveDetector;       //活体检测
seeta::FaceLandmarker *faceMaskDetector;     //五官遮挡检测
seeta::AgePredictor *agePredictor;           //年龄预测
seeta::GenderPredictor *genderPredictor;     //性别评估
seeta::MaskDetector *maskDetector;           //口罩检测
seeta::EyeStateDetector *eyeStateDetector;   //眼睛状态检测
seeta::QualityRule *qualityClarity;          //清晰度评估(传统)
seeta::QualityRule *qualityBright;           //明亮度评估(传统)
seeta::QualityRule *qualityResolution;       //分辨率评估
seeta::QualityOfPose *qualityPose;           //姿态评估(传统)
seeta::QualityOfIntegrity *qualityIntegrity; //完整性评估
seeta::FaceRecognizer *faceRecognizer;       //人脸识别faceFecognition
seeta::FaceTracker *faceTracker;             //人脸跟踪器

static string model_path = "./model";
static SeetaDevice seetaDevice = SEETA_DEVICE_AUTO; //SEETA_DEVICE_AUTO = 0,SEETA_DEVICE_CPU  = 1,SEETA_DEVICE_GPU  = 2
static int deviceId = 0;                            //使用gpu 的id
static int track_width = 1920;                      //人脸跟踪视频的分辨率宽
static int track_height = 1080;                     //人脸跟踪视频的分辨率高

seeta::FaceDetector *new_fd()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/face_detector.csta");
    //按调用文件所在的工作目录文件所在的路径为参考,而不是动态库所在路径.但如果是执行文件,那么按执行文件的位置
    return new seeta::FaceDetector(setting);
}

// 特征点5 检测器
seeta::FaceLandmarker *new_ld5()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/face_landmarker_pts5.csta");
    return new seeta::FaceLandmarker(setting);
}

// 特征点68 检测器
seeta::FaceLandmarker *new_ld68()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/face_landmarker_pts68.csta");
    return new seeta::FaceLandmarker(setting);
}
// 活体检测
seeta::FaceAntiSpoofing *new_live()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/fas_first.csta");
    setting.append(model_path + "/fas_second.csta");
    return new seeta::FaceAntiSpoofing(setting);
}
////五官遮挡检测
seeta::FaceLandmarker *new_facemask()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/face_landmarker_mask_pts5.csta");
    return new seeta::FaceLandmarker(setting);
}
////性别检测
seeta::GenderPredictor *new_genderP()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/gender_predictor.csta");
    return new seeta::GenderPredictor(setting);
};
////年龄检测
seeta::AgePredictor *new_ageP()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/age_predictor.csta");
    return new seeta::AgePredictor(setting);
}
//////眼睛状态检测
seeta::EyeStateDetector *new_eyeD()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/eye_state.csta");
    return new seeta::EyeStateDetector(setting);
}
////口罩检测
seeta::MaskDetector *new_maskD()
{
    seeta::ModelSetting setting;
    setting.device = seetaDevice; //GPU CPU AUTO
    setting.id = deviceId;
    setting.append(model_path + "/mask_detector.csta");
    return new seeta::MaskDetector(setting);
}
//人脸跟踪
seeta::FaceTracker *new_ft()
{
    seeta::ModelSetting setting;
    setting.append(model_path + "/face_detector.csta");
    return new seeta::FaceTracker(setting, track_width, track_height);
}
//人脸识别
seeta::FaceRecognizer *new_fr()
{
    seeta::ModelSetting setting;
    setting.append(model_path + "/face_recognizer.csta");
    return new seeta::FaceRecognizer(setting);
}

// 模型加载
void Init_face()
{
    faceDetector = new_fd(); //人脸框检测
}
void Init_land5()
{
    landDetector5 = new_ld5(); //5特征点检测
}
void Init_land68()
{
    landDetector68 = new_ld68(); //68特征点检测
}
void Init_live()
{
    liveDetector = new_live(); //活体检测
}
void Init_faceMask()
{
    faceMaskDetector = new_facemask(); //五官遮挡检测
}
void Init_age()
{
    agePredictor = new_ageP(); //年龄检测
}
void Init_gender()
{
    genderPredictor = new_genderP(); //性别检测
}
void Init_mask()
{
    maskDetector = new_maskD(); //口罩检测
}
void Init_eyeState()
{
    eyeStateDetector = new_eyeD(); //眼睛状态检测
}

void Init_clarity()
{
    qualityClarity = new seeta::QualityOfClarity(); //清晰度评估(传统)
}
void Init_bright()
{
    qualityBright = new seeta::QualityOfBrightness(); //明亮度评估(传统)
}
void Init_resolution()
{
    qualityResolution = new seeta::QualityOfResolution(); //分辨率评估(传统)
}
void Init_pose()
{
    qualityPose = new seeta::QualityOfPose(); //姿态评估(传统)
}
void Init_integrity()
{
    qualityIntegrity = new seeta::QualityOfIntegrity(); //人脸完整性评估
}
void Init_faceTrack()
{
    faceTracker = new_ft();
}

void Init_recognition()
{
    faceRecognizer = new_fr(); //人脸识别
}
int Init(int func)
{
    if (func == 0)
    {
        Init_face();
        cout << "初始化人脸检测" << endl;
        return 1;
    }
    else if (func == 1)
    {
        Init_land5();
        cout << "初始化人脸5特征点检测" << endl;

        return 1;
    }
    else if (func == 2)
    {
        Init_land68();
        cout << "初始化人脸68特征点检测" << endl;

        return 1;
    }
    else if (func == 3)
    {
        Init_live(); //成功
        cout << "初始化活体检测" << endl;
        return 1;
    }
    else if (func == 4)
    {
        Init_faceMask(); //成功
        cout << "初始化五官遮挡检测" << endl;
        return 1;
    }
    else if (func == 5)
    {
        Init_age(); //成功
        cout << "初始化年龄评估" << endl;
        return 1;
    }
    else if (func == 6)
    {
        Init_gender(); //成功
        cout << "初始化性别评估" << endl;
        return 1;
    }
    else if (func == 7)
    {
        Init_mask(); //成功
        cout << "初始化口罩检测" << endl;
        return 1;
    }
    else if (func == 8)
    {
        Init_eyeState(); //成功
        cout << "初始化眼睛状态检测" << endl;
        return 1;
    }
    else if (func == 9)
    {
        Init_clarity(); //成功
        cout << "初始化清晰度评估" << endl;

        return 1;
    }
    else if (func == 10)
    {
        Init_bright(); //成功
        cout << "初始化明亮度评估" << endl;

        return 1;
    }
    else if (func == 11)
    {
        Init_resolution(); //成功
        cout << "初始化分辨率评估" << endl;
        return 1;
    }
    else if (func == 12)
    {
        Init_pose(); //成功
        cout << "初始化人脸姿态质量评估" << endl;

        return 1;
    }
    else if (func == 13)
    {
        Init_integrity(); //成功
        cout << "初始化人脸完整度评估" << endl;

        return 1;
    }
    else if (func == 14)
    {
        Init_faceTrack(); //成功
        cout << "初始化人脸跟踪" << endl;

        return 1;
    }
    else if (func == 15)
    {
        Init_recognition(); //人脸识别
        cout << "初始化人脸识别" << endl;

        return 1;
    }
    else
        return 0;
}
// #define DLLEXPORT extern "C" __declspec(dllexport)
//获取模型文件目录
extern "C" void get_modelpath(char *path)
{
    cout << "path:" << path << endl;
    model_path = path;
}

//设置使用gpu还是cpu还是auto,默认为auto
extern "C" void set_device(int device, int id)
{
    if (device == 1)
    {
        seetaDevice = SEETA_DEVICE_CPU;
        cout << "seetaDevice : SEETA_DEVICE_CPU" << endl;
    }
    else if (device == 2)
    {
        seetaDevice = SEETA_DEVICE_GPU;
        cout << "seetaDevice : SEETA_DEVICE_GPU" << endl;
    }
    else
    {
        seetaDevice = SEETA_DEVICE_AUTO;
        cout << "seetaDevice : SEETA_DEVICE_AUTO" << endl;
    }
    deviceId = id;
    cout << "deviceId=" << deviceId << endl;
}
//加载模型 , 求的数组长度错误,所以直接作为参数传入
extern "C" int InitEngine(int model_ids[], int len)
{
    for (int i = 0; i < len; i++)
    {
        int model_id = model_ids[i];
        int init_ret = Init(model_id);
        if (init_ret == 0)
        {
            return 0;
        }
    }
    return 1;
}

//人脸框检测
extern "C" SeetaFaceInfoArray Detect(SeetaImageData &simage)
{
    SeetaFaceInfoArray faces = faceDetector->detect(simage);
    return faces;
}
//人脸检测阈值
extern "C" void SetProperty(int property, float value)
{
    if (property == 1)
    {
        faceDetector->set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, value); //最小人脸尺寸,默认20
    }
    else if (property == 2)
    {
        faceDetector->set(seeta::FaceDetector::PROPERTY_THRESHOLD, value); //置信度阈值
    }
    else if (property == 3)
    {
        faceDetector->set(seeta::FaceDetector::PROPERTY_MAX_IMAGE_WIDTH, value); //可检测的图像最大宽度
    }
    else if (property == 4)
    {
        faceDetector->set(seeta::FaceDetector::PROPERTY_MAX_IMAGE_HEIGHT, value); //可检测的图像最大高度
    }
}
//5特征点检测

extern "C" int mark5(SeetaImageData &simage, SeetaRect &box, SeetaPointF points5[5])
{
    std::vector<SeetaPointF> points = landDetector5->mark(simage, box);
    int size = points.size();
    if (size == 0)
    {
        return 0;
    }
    for (int i = 0; i < size; i++)
        points5[i] = points[i];
    return 1;
}

// extern "C" int test(int)
//68特征点检测
extern "C" int mark68(SeetaImageData &simage, SeetaRect &box, SeetaPointF points68[5])
{
    std::vector<SeetaPointF> points = landDetector68->mark(simage, box);
    int size = points.size();
    if (size == 0)
    {
        return 0;
    }
    for (int i = 0; i < size; i++)
        points68[i] = points[i];
    return 1;
}
extern "C" void SetLiveThreshold(float clarity, float reality)
{
    liveDetector->SetThreshold(clarity, reality);
}
//活体检测-返回是否为活体
extern "C" int Predict(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    //方法1:转为vector,利用.data提取地址
    // std::vector<SeetaPointF> points;
    // for (int i = 0; i < 5; i++)
    // {
    //     points.push_back(points5[i]);
    // }
    // auto status = liveDetector->Predict(simage, box, points.data());
    //方法2:直接提取第一个元素的地址入参
    auto status = liveDetector->Predict(simage, box, &points5[0]);
    int ret = int(status);
    return ret; //=0真实,1攻击,2不确定
}

//活体检测-返回清晰度跟活体值
extern "C" void GetPreFrameScore(float *clarity, float *reality)
{
    liveDetector->GetPreFrameScore(clarity, reality);
}
//视频活体检测
extern "C" int PredictVideo(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    auto status = liveDetector->Predict(simage, box, &points5[0]);
    int ret = int(status);
    return ret; //=0真实,1攻击,2不确定
}
//视频活体检测重置
extern "C" void ResetVideo()
{
    liveDetector->ResetVideo();
}

//五官遮挡检测,
extern "C" int markMask(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5], int face_mask[5])
{
    auto point_masks = faceMaskDetector->mark_v2(simage, box);
    int size = point_masks.size();
    if (size == 0)
    {
        return 0;
    }
    for (int i = 0; i < size; i++)
    {
        points5[i] = point_masks[i].point;
        face_mask[i] = point_masks[i].mask;
    }
    return 1;
}

//年龄检测
extern "C" int PredictAgeWithCrop(SeetaImageData &simage, SeetaPointF points5[5])
{
    int age = 0;
    agePredictor->PredictAgeWithCrop(simage, &points5[0], age);
    return age;
}
extern "C" int PredictAge(SeetaImageData &simage)
{
    int age = 0;
    agePredictor->PredictAge(simage, age);
    return age;
}

//性别检测
extern "C" int PredictGenderWithCrop(SeetaImageData &simage, SeetaPointF points5[5])
{
    seeta::GenderPredictor::GENDER gender;
    int genderOut = 0;
    genderPredictor->PredictGenderWithCrop(simage, &points5[0], gender);
    if (gender == seeta::GenderPredictor::FEMALE)
        genderOut = 1;
    return genderOut;
}
extern "C" int PredictGender(SeetaImageData &simage)
{
    seeta::GenderPredictor::GENDER gender;
    int genderOut = 0;
    genderPredictor->PredictGender(simage, gender);
    if (gender == seeta::GenderPredictor::FEMALE)
        genderOut = 1;
    return genderOut;
}

//口罩检测
extern "C" int DetectMask(SeetaImageData &simage, const SeetaRect &box)
{
    float score = 0;
    bool mask = maskDetector->detect(simage, box, &score);
    return int(mask);
}
//人眼状态检测
extern "C" void DectectEye(SeetaImageData &simage, SeetaPointF points5[5], int states[2])
{
    seeta::EyeStateDetector::EYE_STATE left_eye, right_eye;
    vector<string> EYE_STATE_STR{"close", "open", "random", "unknown"};
    eyeStateDetector->Detect(simage, &points5[0], left_eye, right_eye);
    states[0] = int(left_eye);
    states[1] = int(right_eye);
}

//清晰度评估
extern "C" char *ClarityEvaluate(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    vector<char *> level_string = {"LOW", "MEDIUM", "HIGH"};
    seeta::QualityResult result = qualityClarity->check(simage, box, &points5[0], 5);
    char *level = level_string[result.level];
    return level;
}
//明亮度评估
extern "C" char *BrightEvaluate(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    vector<char *> level_string = {"LOW", "MEDIUM", "HIGH"};
    seeta::QualityResult result = qualityBright->check(simage, box, &points5[0], 5);
    char *level = level_string[result.level];
    return level;
}
//分辨率评估
extern "C" char *ResolutionEvaluate(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    vector<char *> level_string = {"LOW", "MEDIUM", "HIGH"};
    seeta::QualityResult result = qualityResolution->check(simage, box, &points5[0], 5);
    char *level = level_string[result.level];
    return level;
}
//人脸姿态质量评估
extern "C" char *PoseEvaluate(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    vector<char *> level_string = {"LOW", "MEDIUM", "HIGH"};
    seeta::QualityResult result = qualityPose->check(simage, box, &points5[0], 5);
    char *level = level_string[result.level];
    return level;
}
//人脸完整性评估
extern "C" char *IntegrityEvaluate(SeetaImageData &simage, const SeetaRect &box, SeetaPointF points5[5])
{
    vector<char *> level_string = {"LOW", "MEDIUM", "HIGH"};
    seeta::QualityResult result = qualityIntegrity->check(simage, box, &points5[0], 5);
    char *level = level_string[result.level];
    return level;
}
//人脸跟踪
extern "C" SeetaTrackingFaceInfoArray Track(SeetaImageData &simage)
{
    SeetaTrackingFaceInfoArray cfaces = faceTracker->Track(simage);
    return cfaces;
}
//人脸跟踪清除
extern "C" void Reset()
{
    faceTracker->Reset();
}
//人脸跟踪最小尺寸设置
extern "C" void SetMinFaceSize(int facesize)
{
    faceTracker->SetMinFaceSize(facesize);
}
//人脸跟踪器的分辨率设置
extern "C" void SetTrackResolution(int width, int height)
{
    track_width = width;
    track_height = height;
}
//人脸跟踪人脸置信度设置
extern "C" void SetThreshold(float threshold)
{
    faceTracker->SetThreshold(threshold);
}
//人脸跟踪检测间隔
extern "C" void SetInterval(int interval)
{
    faceTracker->SetInterval(interval);
}
//人脸跟踪线程数设置
extern "C" void SetSingleCalculationThreads(int num)
{
    faceTracker->SetSingleCalculationThreads(num);
}
//裁剪人脸
extern "C" SeetaImageData CropFace(SeetaImageData &simage, SeetaPointF points5[5])
{
    SeetaImageData face = faceRecognizer->CropFaceV2(simage, &points5[0]);
    return face;
}
//裁剪图特征提取
extern "C" int ExtractCroppedFace(SeetaImageData &simage, float feature[1024])
{
    faceRecognizer->ExtractCroppedFace(simage, &feature[0]);
    return 1;
}
//原图提取特征
extern "C" int Extract(SeetaImageData &simage, SeetaPointF points5[5], float feature[1024])
{
    faceRecognizer->Extract(simage, &points5[0], &feature[0]);
    return 1;
}
//相似度计算
extern "C" float CalculateSimilarity(float feature1[1024], float feature2[1024])
{
    float score = faceRecognizer->CalculateSimilarity(&feature1[0], &feature2[0]);
    return score;
}
// int main(int argc, char const *argv[])
// {
//     cout << "请输入图片地址:" << endl;
//     string picpath;
//     cin >> picpath;

//     // const string picpath = "/home/yangjian/Projects/FaceAPI_demo/images/ch1.jpg";
//     cv::Mat image = cv::imread(picpath, cv::IMREAD_COLOR); //第二个参数IMREA_COLOR让获取的图像数据是连续存储的
//     if (image.empty())
//     {
//         return -1;
//     }

//     SeetaImageData simage;
//     simage.width = image.cols;
//     simage.height = image.rows;
//     simage.channels = image.channels();
//     simage.data = image.data;

//     seeta::ModelSetting setting;
//     setting.device = SEETA_DEVICE_AUTO; //GPU CPU AUTO
//     setting.id = 0;
//     setting.append("../model/face_detector.csta");
//     //按原始cpp文件所在的路径为参考,而不是动态库所在路径.但如果是执行文件,那么按执行文件的位置
//     seeta::FaceDetector faceDetector(setting);
//     SeetaFaceInfoArray faces = faceDetector.detect(simage);

//     for (int i = 0; i < faces.size; i++)
//     {
//         auto face = faces.data[i];
//         SeetaRect box = face.pos;
//         float score = face.score;
//         cout << "x:" << box.x << " y:" << box.y << " w:" << box.width << " h:" << box.height << " score:" << score << endl;
//     }
//     return 0;
// }