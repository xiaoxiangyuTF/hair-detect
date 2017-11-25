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
	//����LBPH���к��У�grid_x*grid_y��ʾ��ͼ��ָ����ôЩ�飬numPatterns��ʾLBPֵ��ģʽ����
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	if (src.empty())
	{
		return result.reshape(1, 1);
	}
	int resultRowIndex = 0;
	//��ͼ����зָ�ָ��grid_x*grid_y�飬grid_x��grid_yĬ��Ϊ8
	for (int i = 0; i<grid_x; i++)
	{
		for (int j = 0; j<grid_y; j++)
		{
			//ͼ��ֿ�
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			//����ֱ��ͼ
			Mat hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), true);
			//��ֱ��ͼ�ŵ�result��
			Mat rowResult = result.row(resultRowIndex);
			hist_cell.reshape(1, 1).convertTo(rowResult, CV_32FC1);
			resultRowIndex++;
		}
	}
	return result.reshape(1, 1);
}

Mat getLocalRegionLBPH(const Mat & src, int minValue, int maxValue, bool normed)
{
	//����洢ֱ��ͼ�ľ���
	cv::Mat result;
	//����õ�ֱ��ͼbin����Ŀ��ֱ��ͼ����Ĵ�С
	int histSize = maxValue - minValue + 1;
	//����ֱ��ͼÿһά��bin�ı仯��Χ
	float range[] = { static_cast<float>(minValue),static_cast<float>(maxValue + 1) };
	//����ֱ��ͼ����bin�ı仯��Χ
	const float* ranges = { range };
	//����ֱ��ͼ��src��Ҫ����ֱ��ͼ��ͼ��1��Ҫ����ֱ��ͼ��ͼ����Ŀ��0�Ǽ���ֱ��ͼ���õ�ͼ���ͨ����ţ���0����
	//Mat()��Ҫ�õ���ģ��resultΪ�����ֱ��ͼ��1Ϊ�����ֱ��ͼ��ά�ȣ�histSizeֱ��ͼ��ÿһά�ı仯��Χ
	//ranges������ֱ��ͼ�ı仯��Χ�������յ㣩
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &ranges, true, false);
	//��һ��
	
	if (normed)
	{
		result /= (float)src.total();
	}
	//��ʾֱ��ͼ
	/*Mat histimg = getHistImg(result);
	Mat pic50;
	Size dsize = { 256,256 };//����ͼ��Ϊ48*48
	resize(histimg, pic50, dsize, 0, 0, INTER_LINEAR);
	namedWindow("ok3", CV_WINDOW_AUTOSIZE);
	imshow("ok3", pic50);
	waitKey(0);*/
	//�����ʾ��ֻ��1�еľ���
	return result.reshape(1, 1);
}

Mat getHistImg(const MatND& hist)
{
	double maxVal = 0;
	double minVal = 0;

	//�ҵ�ֱ��ͼ�е����ֵ����Сֵ
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	int histSize = hist.rows;
	Mat histImg(histSize, histSize, CV_8U, Scalar(255));
	// ��������ֵΪͼ��߶ȵ�90%
	int hpt = static_cast<int>(0.9*histSize);

	for (int h = 0; h<histSize; h++)
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histImg, Point(h, histSize), Point(h, histSize - intensity), Scalar::all(0));
	}

	return histImg;
}

