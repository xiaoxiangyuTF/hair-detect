#include "stdafx.h"
#include<iostream>
#include <opencv2/opencv.hpp>  
#include <contrib\contrib.hpp>
#include "LBP.h"


#include <highgui.hpp>
#include <imgproc\imgproc.hpp>  
#include"TestSvm.h"
using namespace std;
using namespace cv;

int main()
{
	/*测试OPENCV运行*/
	/*
	Mat pic = imread("F:\\VS_code\\hair detect\\nature\\1.jpg");
	namedWindow("ok", CV_WINDOW_AUTOSIZE);
	imshow("ok",pic);
	waitKey(0);
	*/

	cv::Directory dir1, dir2; 
	/*数据文件路径*/
	string train_pos_data_path = "F:\\VS_code\\hair detect\\Patch1k\\Hair\\Training\\";
	string train_neg_data_path = "F:\\VS_code\\hair detect\\Patch1k\\NonHair\\Training\\";
	string test_pos_data_path = "F:\\VS_code\\hair detect\\Patch1k\\Hair\\Testing\\";
	string test_neg_data_path = "F:\\VS_code\\hair detect\\Patch1k\\NonHair\\Testing\\";
	const char classifierSavePath[256] = "F:\\VS_code\\hair detect\\SVM_model2.txt";
	string exten1 = "*.jpg";
	bool addPath1 = false;//true 
	int num, k;
	num = k = 0;
	float accuracy;
	//vector<string> train_neg_filenames = dir.GetListFilesR(train_neg_data_path, exten1, addPath1);
	//vector<string> test_pos_filenames = dir.GetListFilesR(test_pos_data_path, exten1, addPath1);
	//vector<string> test_neg_filenames = dir.GetListFilesR(test_neg_data_path, exten1, addPath1);
	/*HOG特征参数定义*/
	HOGDescriptor *hog = new HOGDescriptor(cvSize(48, 48), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> hog_descriptors;

	/*读取训练数据+提取特征*/
	vector<string> pdata_filenames, ndata_filenames;
	pdata_filenames = dir1.GetListFilesR(train_pos_data_path, exten1, addPath1); //获取训练正样本的数据文件名
	ndata_filenames = dir2.GetListFilesR(train_neg_data_path, exten1, addPath1);
	int TotalSampleNum = pdata_filenames.size() + ndata_filenames.size();
	CvMat *DataFeatureMat = cvCreateMat(TotalSampleNum, 959, CV_32FC1);
	CvMat *DataLabelMat = cvCreateMat(TotalSampleNum, 1, CV_32FC1);
	Mat pic, pic_48, pic_gray, hist;
	for (int i = 0; i < pdata_filenames.size(); i++)
	{
		string path = train_pos_data_path + pdata_filenames[num++]; //字符串是可以直接相加的...忘记了..
		pic = imread(path);	//读取图像
		Size dsize = { 48,48 };//调整图像为48*48
		resize(pic, pic_48, dsize, 0, 0, INTER_LINEAR);
		hog->compute(pic_48, hog_descriptors, Size(48, 48), Size(0, 0));
		for (int j = 0; j < hog_descriptors.size(); j++)
			CV_MAT_ELEM(*DataFeatureMat, float, i, j) = hog_descriptors[j];
		cvtColor(pic_48, pic_gray, CV_BGR2GRAY);
		hist = getLBPH(UniformLBP(pic_gray), 59, 1, 1, 1);
		k = 0;
		for (int j = hog_descriptors.size(); j < (hog_descriptors.size() + hist.cols); j++)
			CV_MAT_ELEM(*DataFeatureMat, float, i, j) = hist.at<float>(0, k++);
		CV_MAT_ELEM(*DataLabelMat, float, i, 0) = 1;
	}	
	num = 0;
	for (int i = pdata_filenames.size(); i < (pdata_filenames.size() + ndata_filenames.size()); i++)
	{		
		string path = train_neg_data_path + ndata_filenames[num++];
		pic = imread(path);	//读取图像
		Size dsize = { 48,48 };//调整图像为48*48
		resize(pic, pic_48, dsize, 0, 0, INTER_LINEAR);
		hog->compute(pic_48, hog_descriptors, Size(48, 48), Size(0, 0));
		for (int j = 0; j < hog_descriptors.size(); j++)
		{
			CV_MAT_ELEM(*DataFeatureMat, float, i, j) = hog_descriptors[j];
		}
		cvtColor(pic_48, pic_gray, CV_BGR2GRAY);
		hist = getLBPH(UniformLBP(pic_gray), 59, 1, 1, 1);
		k = 0;
		for (int j = hog_descriptors.size(); j < (hog_descriptors.size() + hist.cols); j++)
		{
			CV_MAT_ELEM(*DataFeatureMat, float, i, j) = hist.at<float>(0, k++);
		}
		CV_MAT_ELEM(*DataLabelMat, float, i, 0) = -1;
	}
	
	int svm_type = CvSVM::C_SVC;
	int kernel_type = CvSVM::POLY;
	float gamma = 0.07;
	float degree = 3;
	float C = 0.1;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.1);//FLT_EPSILON是最接近于0的浮点数
	CvSVMParams myparams = CvSVMParams(CvSVM::C_SVC, CvSVM::POLY, degree, gamma, 1, C, 0, 0, NULL, criteria);
	CvSVM mysvm;
	mysvm.train(DataFeatureMat, DataLabelMat, NULL, NULL, myparams); //用SVM线性分类器训练
	mysvm.save(classifierSavePath);
	
	
	

	accuracy = TestSvm(test_pos_data_path, test_neg_data_path, classifierSavePath);
	cout << "the accuracy = " << accuracy << endl;



	return 0;
}