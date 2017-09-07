#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/layers/data_layer.hpp"
using namespace caffe;
namespace caffe{
	extern INSTANTIATE_CLASS(DataLayer);
}

int main(int argc, char** argv)
{
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::Net<float> net("F:/tmp/my_test.prototxt", caffe::TEST);
	for (int i = 0; i < 1; i++)
	{
		net.Forward();
	}

	return 0;
}

