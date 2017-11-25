#include "stdafx.h"
#include"LBP.h"

Mat UniformLBP(Mat img)
{
	uchar UTable[256];
	memset(UTable, 0, 256 * sizeof(uchar));
	uchar temp = 1;
	for (int i = 0; i<256; i++)
	{
		if (getHopCount(i) <= 2)
		{
			UTable[i] = temp;
			++temp;
		}
	}
	Mat result;
	result.create(img.rows - 2, img.cols - 2, img.type());

	result.setTo(0);

	for (int i = 1; i<img.rows - 1; i++)
	{
		for (int j = 1; j<img.cols - 1; j++)
		{
			uchar center = img.at<uchar>(i, j);
			uchar code = 0;
			code |= (img.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (img.at<uchar>(i - 1, j) >= center) << 6;
			code |= (img.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (img.at<uchar>(i, j + 1) >= center) << 4;
			code |= (img.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (img.at<uchar>(i + 1, j) >= center) << 2;
			code |= (img.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (img.at<uchar>(i, j - 1) >= center) << 0;
			result.at<uchar>(i - 1, j - 1) = UTable[code];
		}
	}
	return result;
}

int getHopCount(uchar i)
{
	uchar a[8] = { 0 };
	int cnt = 0;
	int k = 7;

	while (k)
	{
		a[k] = i & 1;
		i = i >> 1;
		--k;
	}

	for (int k = 0; k<7; k++)
	{
		if (a[k] != a[k + 1])
			++cnt;
	}

	if (a[0] != a[7])
		++cnt;

	return cnt;
}

Mat getLBPH(InputArray _src, int numPatterns, int grid_x, int grid_y, bool normed)
{
	Mat src = _src.getMat();
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;
	//定义LBPH的行和列，grid_x*grid_y表示将图像分割成这么些块，numPatterns表示LBP值的模式种类
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	if (src.empty())
	{
		return result.reshape(1, 1);
	}
	int resultRowIndex = 0;
	//对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
	for (int i = 0; i<grid_x; i++)
	{
		for (int j = 0; j<grid_y; j++)
		{
			//图像分块
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			//计算直方图
			Mat hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), true);
			//将直方图放到result中
			Mat rowResult = result.row(resultRowIndex);
			hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
			resultRowIndex++;
		}
	}
	return result.reshape(1, 1);
}

Mat getLocalRegionLBPH(const Mat & src, int minValue, int maxValue, bool normed)
{
	//定义存储直方图的矩阵
	cv::Mat result;
	//计算得到直方图bin的数目，直方图数组的大小
	int histSize = maxValue - minValue + 1;
	//定义直方图每一维的bin的变化范围
	float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
	//定义直方图所有bin的变化范围
	const float* ranges = { range };
	//计算直方图，src是要计算直方图的图像，1是要计算直方图的图像数目，0是计算直方图所用的图像的通道序号，从0索引
	//Mat()是要用的掩模，result为输出的直方图，1为输出的直方图的维度，histSize直方图在每一维的变化范围
	//ranges，所有直方图的变化范围（起点和终点）
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &ranges, true, false);
	//归一化
	
	if (normed)
	{
		result /= (float)src.total();
	}
	//显示直方图
	/*Mat histimg = getHistImg(result);
	Mat pic50;
	Size dsize = { 256,256 };//调整图像为48*48
	resize(histimg, pic50, dsize, 0, 0, INTER_LINEAR);
	namedWindow("ok3", CV_WINDOW_AUTOSIZE);
	imshow("ok3", pic50);
	waitKey(0);*/
	//结果表示成只有1行的矩阵
	return result.reshape(1, 1);
}

Mat getHistImg(const MatND& hist)
{
	double maxVal = 0;
	double minVal = 0;

	//找到直方图中的最大值和最小值
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	int histSize = hist.rows;
	Mat histImg(histSize, histSize, CV_8U, Scalar(255));
	// 设置最大峰值为图像高度的90%
	int hpt = static_cast<int>(0.9*histSize);

	for (int h = 0; h<histSize; h++)
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histImg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
	}

	return histImg;
}

