#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace caffe;

int main(int argc, char** argv)
{
	Net<float> templ_feature_net("F:/Tracking/SINT/Sim-fc_cpp/matcaffe/matcaffe/tmpl.prototxt", TEST);
	templ_feature_net.CopyTrainedLayersFromBinaryProto("F:/Tracking/SINT/Sim-fc_cpp/matcaffe/matcaffe/params.caffemodel");

	Net<float> compare_net("F:/Tracking/SINT/Sim-fc_cpp/matcaffe/matcaffe/compare.prototxt", TEST);
	compare_net.CopyTrainedLayersFromBinaryProto("F:/Tracking/SINT/Sim-fc_cpp/matcaffe/matcaffe/params.caffemodel");


	cv::Mat tmpl, image;
	tmpl = cv::imread("F:/Tracking/SINT/Sim-fc_cpp/matcaffe/matcaffe/templ.png");
	cv::resize(tmpl, tmpl, cv::Size(127, 127));
	cv::cvtColor(tmpl, tmpl, CV_RGB2BGR);
	tmpl = tmpl.t();
	tmpl.convertTo(tmpl, CV_32FC3);
	
	image = cv::imread("F:/Tracking/SINT/Sim-fc_cpp/matcaffe/matcaffe/00000001.jpg");
	cv::resize(image, image, cv::Size(383, 383));
	cv::cvtColor(image, image, CV_RGB2BGR);
	image = image.t();

	image.convertTo(image, CV_32FC3);
	std::vector<cv::Mat> templ_channels, image_channels;
	
	cv::split(tmpl, templ_channels);


	cv::split(image, image_channels);



	templ_feature_net.blob_by_name("data").get()->Reshape(1, 3, 127, 127);
	float * tmpl_data = templ_feature_net.blob_by_name("data").get()->mutable_cpu_data();
	
	for (int i = 0; i < templ_channels.size(); i++)
	{
		cv::Mat channel(127, 127, CV_32FC1, tmpl_data);
		templ_channels[i].copyTo(channel);
		tmpl_data += 127 * 127;
	}
	tmpl_data = templ_feature_net.blob_by_name("data").get()->mutable_cpu_data();

	templ_feature_net.Forward();
	const float *templ_feature_data = templ_feature_net.blob_by_name("conv5").get()->cpu_data();
	

	std::vector<int> templ_feature_map_shape =
		templ_feature_net.blob_by_name("conv5").get()->shape();

	compare_net.blob_by_name("target_feature_map").get()->Reshape(templ_feature_map_shape);

	int N = templ_feature_map_shape[0];
	for (int i = 1; i < templ_feature_map_shape.size();i++)
	{
		N *= templ_feature_map_shape[i];
	}
	
	float *templ_feature_data_input =
		compare_net.blob_by_name("target_feature_map").get()->mutable_cpu_data();

	caffe_copy(N, templ_feature_data, templ_feature_data_input);

	compare_net.blob_by_name("data").get()->Reshape(1, 3, 383, 383);
	float * image_data = compare_net.blob_by_name("data").get()->mutable_cpu_data();

	for (int i = 0; i < templ_channels.size(); i++)
	{
		cv::Mat channel(383, 383, CV_32FC1, image_data);
		image_channels[i].copyTo(channel);
		image_data += 383 * 383;
	}
	
	compare_net.Forward();
	
	int resp_H = compare_net.blob_by_name("xcorr").get()->shape(2);
	int resp_W = compare_net.blob_by_name("xcorr").get()->shape(3);
	float* response_map_data = compare_net.blob_by_name("xcorr").get()->mutable_cpu_data();
	cv::Mat response_map(resp_W, resp_H, CV_32FC1, response_map_data);


	// FOR DEBUG:
	std::vector<int> target_conv5_shape = templ_feature_map_shape;
	std::vector<int> image_conv5_shape = 
		compare_net.blob_by_name("conv5").get()->shape();

	float *target_conv5_data = templ_feature_net.blob_by_name("conv5").get()->mutable_cpu_data();
	float *image_conv5_data = compare_net.blob_by_name("conv5").get()->mutable_cpu_data();
	std::vector<cv::Mat> target_conv5_channels, image_conv5_channels;

	for (int i = 0; i < target_conv5_shape[1]; i++)
	{
		cv::Mat channel(target_conv5_shape[2], target_conv5_shape[3], CV_32FC1, target_conv5_data);
		target_conv5_channels.push_back(channel);
		target_conv5_data += target_conv5_shape[2] * target_conv5_shape[3];
	}

	for (int i = 0; i < image_conv5_shape[1]; i++)
	{
		cv::Mat channel(image_conv5_shape[2], image_conv5_shape[3], CV_32FC1, image_conv5_data);
		image_conv5_channels.push_back(channel);
		image_conv5_data += image_conv5_shape[2] * image_conv5_shape[3];
	}

	return 0;
}