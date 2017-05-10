#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace caffe;

int main(int argc, char** argv)
{
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::Net<float> net("D:/dataset/COCO/training/experiments/test.prototxt",caffe::TRAIN);

	net.Forward();
	return 0;
}