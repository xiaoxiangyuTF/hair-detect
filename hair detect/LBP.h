
#include "stdafx.h"
#include<iostream>
#include <opencv2/opencv.hpp>  
#include <contrib\contrib.hpp>
#include<opencv2/highgui/highgui.hpp>  
using namespace std;
using namespace cv;

Mat UniformLBP(Mat img);
int getHopCount(uchar i);

Mat getLBPH(InputArray _src, int numPatterns, int grid_x, int grid_y, bool normed);
Mat getLocalRegionLBPH(const Mat& src, int minValue, int maxValue, bool normed);
Mat getHistImg(const MatND& hist);