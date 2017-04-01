#include "Siamesefc.h"
#include "recttools.hpp"


static float subPixelPeak(float left, float center, float right)
{
	float divisor = 2 * center - right - left;

	if (divisor == 0)
		return 0;

	return 0.5 * (right - left) / divisor;
}

SiameseFC::SiameseFC()
{
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	
	this->features_extract_net_.reset();
	this->compare_net_.reset();

SCALES.push_back(0.97);
	SCALES.push_back(1);
SCALES.push_back(1.03);
}


cv::Rect SiameseFC::update(cv::Mat image)
{
	float cx = roi_.x + roi_.width / 2.0;
	float cy = roi_.y + roi_.height / 2.0;


	double pvmax = DBL_MIN;
	cv::Point2i pi_max;
	float scale = 1.0;
	cv::Mat max_response_map;
	for (int idx = 0; idx < SCALES.size(); idx ++)
	{
		float new_width = roi_.width * PADDING * SCALES[idx];
		float new_height = roi_.height * PADDING* SCALES[idx];

		cv::Rect  search_roi(cx - new_width / 2.0, cy - new_height / 2.0,
			new_width, new_height);
		
		cv::Mat search = RectTools::subwindow(image,
			search_roi, cv::BORDER_REPLICATE);

		prepareInputs(search, 
			image_input_blob_->mutable_cpu_data(), image_input_size_);

		compare_net_->Forward();

		cv::Mat response_map(response_size_, 
			CV_32FC1, response_blob_->mutable_cpu_data());

		cv::Point2i pi;
		double pv;
		cv::minMaxLoc(response_map, NULL, &pv, NULL, &pi);
		if (pvmax < pv)
		{
			pvmax = pv;
			scale = SCALES[idx];
			pi_max = pi;
			max_response_map = response_map;
		}


	}

	float peak_value = (float)pvmax;

	//sub-pixel peak estimation, coordinates will be non-integer
	cv::Point2f p((float)pi_max.x, (float)pi_max.y);

	if (pi_max.x > 0 && pi_max.x < max_response_map.cols - 1) {
		p.x += subPixelPeak(max_response_map.at<float>(pi_max.y, pi_max.x - 1),
			peak_value, max_response_map.at<float>(pi_max.y, pi_max.x + 1));
	}

	if (pi_max.y > 0 && pi_max.y < max_response_map.rows - 1) {
		p.y += subPixelPeak(max_response_map.at<float>(pi_max.y - 1, pi_max.x),
			peak_value, max_response_map.at<float>(pi_max.y + 1, pi_max.x));
	}

	p.x -= (max_response_map.cols) / 2;
	p.y -= (max_response_map.rows) / 2;

	cx += p.x * (roi_.width  / response_size_.width * scale);
	cy += p.y* (roi_.height  / response_size_.height * scale);
	float new_width = roi_.width * scale; float new_height = roi_.height * scale;

	roi_ = cv::Rect_<float>(cx - new_width / 2.0, 
		cy - new_height / 2.0, new_width, new_height);
	
	return roi_;
}




void SiameseFC::init(const cv::Rect &roi, cv::Mat image)
{
	if (features_extract_net_ == nullptr || compare_net_ == nullptr)
	{
		std::cerr << "the net should be setup. \n";
		return;
	}
	
	float cx = roi.x + roi.width / 2.0;
	float cy = roi.y + roi.height / 2.0;

	float new_width = roi.width * 1.5;
	float new_height = roi.height * 1.5;

	cv::Rect tmpl_roi_with_context(cx - new_width / 2.0, cy - new_height / 2.0, 
		new_width, new_height);

	cv::Mat tmpl = RectTools::subwindow(image, 
		tmpl_roi_with_context, cv::BORDER_REPLICATE);


	float * tmpl_data = target_input_blob_->mutable_cpu_data();

	prepareInputs(tmpl, tmpl_data, target_input_size_);

	features_extract_net_->Forward();

	target_feature_blob_.reset(features_extract_net_->blob_by_name("conv5").get());

	float *target_feature_data = compare_net_
		->blob_by_name("target_feature_map").get()->mutable_cpu_data();
	
	caffe::caffe_copy(target_feature_blob_->count(0), target_feature_blob_->cpu_data(),
		target_feature_data);

	roi_ = roi;
}




void SiameseFC::setupModel(const std::string &tmpl_proto, 
	const std::string &compare_proto, 
	const std::string &model)
{
	this->features_extract_net_.reset(
		new caffe::Net<float>(tmpl_proto, caffe::TEST)
		);

	this->compare_net_.reset(
		new caffe::Net<float>(compare_proto, caffe::TEST)
		);

	this->features_extract_net_->
		CopyTrainedLayersFromBinaryProto(model);

	this->compare_net_->
		CopyTrainedLayersFromBinaryProto(model);
	
	features_extract_net_->Reshape();
	compare_net_->Reshape();
	target_input_blob_.reset( features_extract_net_->blob_by_name("data").get());
	target_input_size_.height = target_input_blob_->shape(2);
	target_input_size_.width = target_input_blob_->shape(3);

	image_input_blob_.reset(compare_net_->blob_by_name("data").get());
	image_input_size_.height = image_input_blob_->shape(2);
	image_input_size_.width = image_input_blob_->shape(3);

	response_blob_.reset(compare_net_->blob_by_name("xcorr").get());
	response_size_ = cv::Size(response_blob_->shape(3),
		response_blob_->shape(2));

}

void SiameseFC::prepareInputs(cv::Mat img, 
	float *input, 
	const cv::Size &input_size)
{

	cv::resize(img, img, input_size);
	cv::cvtColor(img, img, CV_RGB2BGR);
	//img = img.t();
	img.convertTo(img, CV_32FC3);

	std::vector<cv::Mat> channels;
	cv::split(img, channels);
	for (int i = 0; i < channels.size(); i++)
	{
		cv::Mat channel(input_size.width, input_size.height, CV_32FC1, input);
		channels[i].copyTo(channel);
		input += input_size.area();
	}
}

