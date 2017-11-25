#include "stdafx.h"
#include"TestSvm.h"
float TestSvm(string pospath, string negpath,const char *classifierSavePath)
{
	float accuracy;
	int predict_result;
	int correct_flag = 0;
	CvSVM mysvm;
	mysvm.load(classifierSavePath);
	cv::Directory dir1, dir2;
	string exten1 = "*.jpg";
	bool addPath1 = false;//true 
	int num, k;
	num = k = 0;
	//vector<string> train_neg_filenames = dir.GetListFilesR(train_neg_data_path, exten1, addPath1);
	//vector<string> test_pos_filenames = dir.GetListFilesR(test_pos_data_path, exten1, addPath1);
	//vector<string> test_neg_filenames = dir.GetListFilesR(test_neg_data_path, exten1, addPath1);
	/*HOG特征参数定义*/
	HOGDescriptor *hog = new HOGDescriptor(cvSize(48, 48), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> hog_descriptors;

	/*读取训练数据+提取特征*/
	vector<string> pdata_filenames, ndata_filenames;
	pdata_filenames = dir1.GetListFilesR(pospath, exten1, addPath1); //获取训练正样本的数据文件名
	ndata_filenames = dir2.GetListFilesR(negpath, exten1, addPath1);
	int TotalSampleNum = pdata_filenames.size() + ndata_filenames.size();
	CvMat *TestFeatureMat = cvCreateMat(1, 959, CV_32FC1);
	Mat pic, pic_48, pic_gray, hist;

	/*for (int i = 0; i < pdata_filenames.size(); i++)
	{
		string path = pospath + pdata_filenames[num++]; //字符串是可以直接相加的...忘记了..
		pic = imread(path);	//读取图像
		Size dsize = { 48,48 };//调整图像为48*48
		resize(pic, pic_48, dsize, 0, 0, INTER_LINEAR);
		hog->compute(pic_48, hog_descriptors, Size(48, 48), Size(0, 0));
		for (int j = 0; j < hog_descriptors.size(); j++)
			CV_MAT_ELEM(*TestFeatureMat, float, 0, j) = hog_descriptors[j];
		cvtColor(pic_48, pic_gray, CV_BGR2GRAY);
		hist = getLBPH(UniformLBP(pic_gray), 59, 1, 1, 1);
		k = 0;
		for (int j = hog_descriptors.size(); j < (hog_descriptors.size() + hist.cols); j++)
			CV_MAT_ELEM(*TestFeatureMat, float, 0, j) = hist.at<float>(0, k++);
		predict_result = mysvm.predict(TestFeatureMat);
		if (predict_result == 1)
			correct_flag++;
	}*/
	num = 0;
	for (int i = pdata_filenames.size(); i < (pdata_filenames.size() + ndata_filenames.size()); i++)
	{
		string path = negpath + ndata_filenames[num++];
		pic = imread(path);	//读取图像
		Size dsize = { 48,48 };//调整图像为48*48
		resize(pic, pic_48, dsize, 0, 0, INTER_LINEAR);
		hog->compute(pic_48, hog_descriptors, Size(48, 48), Size(0, 0));
		for (int j = 0; j < hog_descriptors.size(); j++)
		{
			CV_MAT_ELEM(*TestFeatureMat, float, 0, j) = hog_descriptors[j];
		}
		cvtColor(pic_48, pic_gray, CV_BGR2GRAY);
		hist = getLBPH(UniformLBP(pic_gray), 59, 1, 1, 1);
		k = 0;
		for (int j = hog_descriptors.size(); j < (hog_descriptors.size() + hist.cols); j++)
		{
			CV_MAT_ELEM(*TestFeatureMat, float, 0, j) = hist.at<float>(0, k++);
		}
		predict_result = mysvm.predict(TestFeatureMat);
		if (predict_result == -1)
			correct_flag++;
	}
	
	accuracy = (float)correct_flag / (float)TotalSampleNum;
	return accuracy;
}