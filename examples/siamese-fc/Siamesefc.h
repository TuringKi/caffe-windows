#ifndef _SIAMESEFC_H_
#define  _SIAMESEFC_H_

#include "caffe/caffe.hpp"

#include <opencv2/opencv.hpp>


class SiameseFC
{
public:
	SiameseFC();
	~SiameseFC(){}


	void setupModel(const std::string &tmpl_proto,
		const std::string &compare_proto,
		const std::string &model);

	// Initialize tracker
	void init(const cv::Rect &roi, cv::Mat image);

	// Update position based on the new frame
	cv::Rect update(cv::Mat image);

private:

	void prepareInputs(cv::Mat img, float *input, const cv::Size &input_size);



	 std::vector<float> SCALES;
	const float PADDING = 6.0;

	std::shared_ptr<caffe::Blob<float>> target_feature_blob_;


	std::shared_ptr<caffe::Blob<float>> target_input_blob_;
	std::shared_ptr<caffe::Blob<float>> image_input_blob_;
	std::shared_ptr<caffe::Blob<float>> response_blob_;

	std::shared_ptr<caffe::Net<float>> features_extract_net_;
	std::shared_ptr<caffe::Net<float>> compare_net_;

	cv::Size target_input_size_;
	cv::Size image_input_size_;
	cv::Size response_size_;

	// for tracker:
	cv::Rect_<float> roi_;
};


#endif //_SIAMESEFC_H_

