#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda.h>
#include "opencv2/core/cuda.hpp"
#include <math.h>
#include "header.h"

//extern void convert_to_gray (cv::Mat& input, cv::Mat& output);
using std::cout;
using std::endl;

int main()
{
	std::string imagePath = "/home/kattous/Programmes/dev-prog/opencv-project/cuda/3/img_src/4.jpg";
	//Read input image from the disk
	cv::Mat input = cv::imread(imagePath,CV_LOAD_IMAGE_COLOR);
	if(input.empty())
	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

	//Create output image
	cv::Mat output(input.rows,input.cols,CV_8UC1);
	//Call the wrapper function
	convert_to_gray(input,output);
	//Show the input and output
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
        cout << std::setprecision(10) << "Temps de traitement GPU [ms]= " << time << endl;
	cv::Mat imgray;
	int64 e1 = cv::getTickCount();
	cv::cvtColor(input, imgray, CV_RGBA2GRAY);
	int64 e2 = cv::getTickCount();
	double t = (e2 - e1)/cv::getTickFrequency();
	cout << "Temps de traitement CPU [ms]= "<< t*1000 <<endl;

	cv::cuda::GpuMat img_src_Gpu, imgray_Gpu;
	/*img_src_Gpu.upload(input);
        if(!img_src_Gpu.data){
	   cout<< "Image not found!"<<endl;
	}*/

	//cv::cuda::cvtColor(img_src_Gpu, imgray_Gpu, CV_RGBA2GRAY,4);
	cv::Mat result_host;
	//img_src_Gpu.download();
	cv::imshow("Input",output);
	cv::imshow("Output",imgray);
	//cv::imshow("imgray_Gpu",imgray_Gpu);
	//Wait for key press
	cv::waitKey();
	return 0;
}
