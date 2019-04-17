#include "Params.h"

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0
#define RAW_SAMPLE -1

// 摄像机模式/图片模式
#define CAMERA_MODE
// #define PIC_MODE

// 是否在达尔文上跑
// 可用 CV_MAJOR_VERSION代替
// #define RUN_ON_DARWIN

// 在摄像机模式获得样本
#define GET_SAMPLE_LIVING

#define IMG_COLS 128
#define IMG_ROWS 128

#undef MODEL_NAME
#define MODEL_NAME "../SvmTrain/model/BigBall/c_svc_with_moment&lbp.xml"

// 开启选项之后
#ifdef GET_SAMPLE_LIVING

#define POS_COUNTER_INIT_NUM 460
#define NEG_COUNTER_INIT_NUM 511
#define RAW_COUNTER_INIT_NUM 0
#define SAVE_PATH "../../BackUpSource/BigBall/Raw/"

int pos_counter = POS_COUNTER_INIT_NUM;
int neg_counter = NEG_COUNTER_INIT_NUM;
int raw_counter = RAW_COUNTER_INIT_NUM;

string GetPath(string save_path, int lable) {
    stringstream t_ss;
    string t_s;

    if (lable == POS_LABLE) {
        save_path += "Pos/";
        t_ss << pos_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;
    }
    else if(lable == NEG_LABLE) {
        save_path += "Neg/";
        t_ss << neg_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;   
    }
    else {
        save_path += "Raw/";
        t_ss << raw_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;   
    }
    return t_s;
}
#endif


int main(int argc, char const *argv[]) {
    // load SVM model
#if CV_MAJOR_VERSION < 3
    CvSVM tester;
    tester.load(MODEL_NAME);
#else
    cv::Ptr<cv::ml::SVM> tester = cv::ml::SVM::load(MODEL_NAME);
#endif
#ifdef CAMERA_MODE
    cv::VideoCapture cp(0);
    cv::Mat frame; 
    int roi_rect_x = 100;
    int roi_rect_y = 100;
    int roi_rect_col = 2*IMG_COLS;
    int roi_rect_row = 2*IMG_ROWS;
    cv::Rect ROI_Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);

    cp >> frame;
    while (frame.empty()) {
        cp >> frame;
    }
    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr << __LINE__ <<"frame empty"<<endl;
            return -1;
        }
#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        cv::Mat ROI = frame(ROI_Rect).clone();
        cv::resize(ROI, ROI, cv::Size(IMG_COLS, IMG_ROWS));
        cv::HOGDescriptor hog_des(Size(IMG_COLS, IMG_ROWS), Size(16,16), Size(8,8), Size(8,8), 9);
        std::vector<float> hog_vec;
        hog_des.compute(ROI, hog_vec);

        for (int j=0; j<6; j++) {
            cv::Mat ROI_l = GetUsedChannel(ROI, j);
            cv::Moments moment = cv::moments(ROI_l, false);

            cv::Mat lbp_mat;
            // cv::resize(t_image_l, t_image_l, cv::Size(30, 30));
            calExtendLBPFeature(ROI_l, Size(16, 16), lbp_mat);
            for (int k=0; k<lbp_mat.cols; k++) {
                hog_vec.push_back(lbp_mat.at<float>(0, k));
            }

            double hu[7];
            cv::HuMoments(moment, hu);
            for (int k=0; k<7; k++) {
                hog_vec.push_back(hu[k]);
            }
            // for (int k=0; k<lbp_vec.cols; k++) {
            //     t_descrip_vec.push_back(lbp_vec.at<uchar>(0, k));
            // }
        }
        // cout<<hog_vec.size()<<endl;
        cv::Mat t(hog_vec);
        cv::Mat hog_vec_in_mat = t.t();
        hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);
#if CV_MAJOR_VERSION < 3
        int lable = (int)tester.predict(hog_vec_in_mat);
        // cout<<tester.predict(hog_vec_in_mat)<<endl;
        if (lable == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
#else
        cv::Mat lable;
        tester->predict(hog_vec_in_mat, lable);
        if (lable.at<float>(0, 0) == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        } 
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
#endif
        
        cv::imshow("frame", frame);
        char key = cv::waitKey(20);
        if (key == 'q') {
            break;
        }
        else if (key == 'p') {
            cv::imwrite(GetPath(SAVE_PATH, POS_LABLE), ROI);
        }
        else if (key == 'n') {
            cv::imwrite(GetPath(SAVE_PATH, NEG_LABLE), ROI);
        }
        else if (key == 'r') {
            cv::imwrite(GetPath(SAVE_PATH, RAW_SAMPLE), frame);
        }
        else if (key == 'i') {
            roi_rect_col -= 15;
            roi_rect_row -= 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'o') {
            roi_rect_col += 15;
            roi_rect_row += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'w') {
            roi_rect_y -= 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 's') {
            roi_rect_y += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'a') {
            roi_rect_x -= 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
        else if (key == 'd') {
            roi_rect_x += 15;
            ROI_Rect = cv::Rect(roi_rect_x, roi_rect_y, roi_rect_col, roi_rect_row);
        }
    }

#endif

#ifdef PIC_MODE
    // load Test Data
    std::vector<cv::Mat> test_image;
    cv::Mat test_data;
    int test_images_num;
    
    HogGetter hog_getter;
    hog_getter.ImageReader_("D:/baseRelate/code/svm_trial/BackUpSource/People/Test/HSample/", "/*.jpg");
    test_data = hog_getter.HogComputter_();
    test_image = hog_getter.raw_images_;
    test_images_num = hog_getter.sample_nums_;
    // GetImage("D:/baseRelate/code/svm_trial/BackUpSource/People/Test/PosSample/", "*.jpg", test_image, test_data);
    // for (auto i = test_image.begin(); i != test_image.end(); i++) {
    //     cv::imshow("233", *i);
    //     cv::waitKey();
    // }
    
    // Compute correct rate
    // cout<<test_images_num<<endl;
    int correct_nums = 0;
    for (int i=0; i<test_images_num; i++) {
        // t = test_data.rowRange(i, i+1);
        cv::Mat t = test_data.row(i).clone();
        // cout<<t.size()<<endl;
        t.convertTo(t, CV_32FC1);
        int predict_lable = (int)tester.predict(t);
        // cout<<predict_lable<<endl;
        correct_nums += predict_lable;
    }
    cout<< correct_nums*1.0/test_images_num<<endl;
#endif
    return 0;
}

// void GetXsCurrectRate(const string& test_set_path, const string& test_image_postfix,
//                     int currect_lable, double& currect_rate) {
//     cv::Mat test_data
//     HogGetter hog_getter;
//     hog_getter.ImageReader_(test_set_path, test_image_postfix);
//     hog_getter.HogComputter_();
// }