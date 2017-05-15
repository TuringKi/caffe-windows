#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace caffe;

int main(int argc, char** argv)
{
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::Net<float> net("F:/HandPose/CPM_Hands/experiments/test.prototxt",caffe::TRAIN);
	for (int i = 0; i < 8; i++)
	{
		net.Forward();
	}
	
	return 0;
}