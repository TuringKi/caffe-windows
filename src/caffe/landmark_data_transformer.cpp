#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include "opencv/cv.hpp"
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/landmark_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template<typename Dtype>
	static void DecodeFloats(const string& data, size_t idx, Dtype* pf, size_t len) {
		memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
	}

	static string DecodeString(const string& data, size_t idx) {
		string result = "";
		int i = 0;
		while (data[idx + i] != 0){
			result.push_back(char(data[idx + i]));
			i++;
		}
		return result;
	}



	template<typename Dtype>
	LandmarkDataTransformer<Dtype>::LandmarkDataTransformer(const TransformationParameter& param,
		Phase phase)
		: param_(param), phase_(phase) {
		// check if we want to use mean_file
		if (param_.has_mean_file()) {
			CHECK_EQ(param_.mean_value_size(), 0) <<
				"Cannot specify mean_file and mean_value at the same time";
			const string& mean_file = param.mean_file();
			if (Caffe::root_solver()) {
				LOG(INFO) << "Loading mean file from: " << mean_file;
			}
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
			data_mean_.FromProto(blob_proto);
		}
		// check if we want to use mean_value
		if (param_.mean_value_size() > 0) {
			CHECK(param_.has_mean_file() == false) <<
				"Cannot specify mean_file and mean_value at the same time";
			for (int c = 0; c < param_.mean_value_size(); ++c) {
				mean_values_.push_back(param_.mean_value(c));
			}
		}
	}

	template<typename Dtype>
	void LandmarkDataTransformer<Dtype>::Transform(const Datum& datum,
		Dtype* transformed_data) {

		// 对齐后的大小，如果没有，则默认为原图大小
		int crop_h = 0;
		int crop_w = 0;
		if (param_.has_crop_size()) {
			crop_h = param_.crop_size();
			crop_w = param_.crop_size();
		}
		if (param_.has_crop_h()) {
			crop_h = param_.crop_h();
		}
		if (param_.has_crop_w()) {
			crop_w = param_.crop_w();
		}



		const string& data = datum.data();
		const int datum_channels = datum.channels() - 1;
		const int datum_height =datum.height();
		const int datum_width =datum.width();

	
		const Dtype scale = param_.scale();
		const bool do_mirror = param_.mirror() && Rand(2);
		const bool has_mean_file = param_.has_mean_file();
		const bool has_uint8 = data.size() > 0;
		const bool has_mean_values = mean_values_.size() > 0;

		CHECK_GT(datum_channels, 0);
		CHECK_GE(datum_height, crop_h);
		CHECK_GE(datum_width, crop_w);

		Dtype* mean = NULL;
		if (has_mean_file) {
			CHECK_EQ(datum_channels, data_mean_.channels());
			CHECK_EQ(datum_height, data_mean_.height());
			CHECK_EQ(datum_width, data_mean_.width());
			mean = data_mean_.mutable_cpu_data();
		}
		if (has_mean_values) {
			CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
				"Specify either 1 mean_value or as many as channels: " << datum_channels;
			if (datum_channels > 1 && mean_values_.size() == 1) {
				// Replicate the mean_value for simplicity
				for (int c = 1; c < datum_channels; ++c) {
					mean_values_.push_back(mean_values_[0]);
				}
			}
		}



		//before any transformation, get the image from datum
		cv::Mat img = cv::Mat::zeros(datum_height, datum_width, CV_8UC3);

		int offset = img.rows * img.cols;
		int dindex;
		Dtype d_element;
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				cv::Vec3b& rgb = img.at<cv::Vec3b>(i, j);
				for (int c = 0; c < 3; c++){
					dindex = c*offset + i*img.cols + j;
					if (has_uint8)
						d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
					else
						d_element = datum.float_data(dindex);
					rgb[c] = d_element;
				}
			}
		}

		std::vector<cv::Point2f> ldmks(5); //2*5

		//cv::Mat meta_img = cv::Mat::zeros(datum_height, datum_width, CV_32FC1);
		int offset3 = 3 * offset;
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 5; ++j) {
					
				dindex = offset3 + i*datum_width + j * sizeof(float);

				float d_element;
				memcpy(&d_element, const_cast<char*>(&data[dindex]), sizeof(float));
				if (i == 0)
					ldmks[j].x = d_element;
				else
					ldmks[j].y = d_element;
				//meta_img.at<float>(i, j) = d_element;
				
			}
		}

		cv::Mat show_img = img.clone();
		for (int p = 0; p < 5; p++) {

			circle(show_img, ldmks[p], 2, cv::Scalar(0, 255, 255), -1);
		}

		int out_width = crop_w ? crop_w : datum_width;
		int out_height = crop_h ? crop_h : datum_height;
		cv::Mat warp_dst;
		FaceAlignment(img,
			ldmks,
			cv::Size(out_width, out_height),
			warp_dst);
	
		

		//存入top_data

		int height = out_height;
		int width = out_width;

		Dtype datum_element;
		std::vector<cv::Mat> v_channels;
		cv::split(warp_dst, v_channels);
		auto img_data = warp_dst.data;
		int top_index, data_index;
		for (int c = 0; c < datum_channels; ++c) {
			auto img_data = v_channels[c].data;
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					data_index = ( h) * width + w;
					if (do_mirror) {
						top_index = (c * height + h) * width + (width - 1 - w);
					}
					else {
						top_index = (c * height + h) * width + w;
					}
					
					datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(img_data[data_index]));
					
				
					if (has_mean_file) {
						transformed_data[top_index] =
							(datum_element - mean[data_index]) * scale;
					}
					else {
						if (has_mean_values) {
							transformed_data[top_index] =
								(datum_element - mean_values_[c]) * scale;
						}
						else {
							transformed_data[top_index] = datum_element * scale;
						}
					}
				}
			}
		}




	}


# define M_PI           3.14159265358979323846  /* pi */
	template <typename Dtype>
	bool caffe::LandmarkDataTransformer<Dtype>::
		FaceAlignment(cv::Mat input_image, 
		const std::vector<cv::Point2f> &face_points, 
		cv::Size size,
		cv::Mat &warp_dst)
	{
		cv::Point2d eye_center((face_points[0].x + face_points[1].x) / 2.0,
			(face_points[0].y + face_points[1].y) / 2.0);

		cv::Point2d mouth_center((face_points[3].x + face_points[4].x) / 2.0,
			(face_points[3].y + face_points[4].y) / 2.0);

		double ec_mc_y = 60.0;
		double tmp = (mouth_center.x - eye_center.x)*(mouth_center.x - eye_center.x) +
			(mouth_center.y - eye_center.y)*(mouth_center.y - eye_center.y);

		double scale = ec_mc_y / sqrt(tmp);

		cv::Point2d center((face_points[0].x + face_points[1].x + face_points[3].x + face_points[4].x) / 4.0,
			(face_points[0].y + face_points[1].y + face_points[3].y + face_points[4].y) / 4.0);

		double angle = atan2(mouth_center.x - eye_center.x, mouth_center.y - eye_center.y) / M_PI* -180.0;

		auto rot_mat = cv::getRotationMatrix2D(center, angle, scale);

		rot_mat.at<double>(0, 2) -= (center.x - size.width / 2.0);
		rot_mat.at<double>(1, 2) -= (center.y - size.height / 2.0);
		warp_dst = cv::Mat::zeros(size.height, size.width, input_image.type());
		cv::warpAffine(input_image, warp_dst, rot_mat, warp_dst.size());

		return true;
	}


	template<typename Dtype>
	void LandmarkDataTransformer<Dtype>::Transform(const Datum& datum,
		Blob<Dtype>* transformed_blob) {
		// If datum is encoded, decoded and transform the cv::image.
		

		

		const int crop_size = param_.crop_size();
		const int crop_w = param_.crop_w();
		const int crop_h = param_.crop_h();
		const int datum_channels = datum.channels() - 1;
		const int datum_height = datum.height();
		const int datum_width = datum.width();

		// Check dimensions.
		const int channels = transformed_blob->channels();
		const int height = transformed_blob->height();
		const int width = transformed_blob->width();
		const int num = transformed_blob->num();

		CHECK_EQ(channels, datum_channels);
		CHECK_LE(height, datum_height);
		CHECK_LE(width, datum_width);
		CHECK_GE(num, 1);

		if (crop_size) {
			CHECK_EQ(crop_size, height);
			CHECK_EQ(crop_size, width);
		} else if (crop_h && crop_w)
		{
			CHECK_EQ(crop_h, height);
			CHECK_EQ(crop_w, width);
		}
		else {
			CHECK_EQ(datum_height, height);
			CHECK_EQ(datum_width, width);
		}


		Dtype* transformed_data = transformed_blob->mutable_cpu_data();
		Transform(datum, transformed_data);
	}

	template<typename Dtype>
	vector<int> LandmarkDataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
		
		const int crop_size = param_.crop_size();
		
		const int datum_channels = datum.channels();
		const int datum_height = datum.height();
		const int datum_width = datum.width();
		// Check dimensions.
		CHECK_GT(datum_channels, 1);
		CHECK_GE(datum_height, crop_size);
		CHECK_GE(datum_width, crop_size);

		int crop_h = 0;
		int crop_w = 0;
		if (param_.has_crop_size()) {
			crop_h = param_.crop_size();
			crop_w = param_.crop_size();
		}
		if (param_.has_crop_h()) {
			crop_h = param_.crop_h();
		}
		if (param_.has_crop_w()) {
			crop_w = param_.crop_w();
		}


		// Build BlobShape.
		vector<int> shape(4);
		shape[0] = 1;
		shape[1] = datum_channels - 1; //只有前三个channel是图像，最后一个channel是关键点坐标
		shape[2] = (crop_h) ? crop_h : datum_height;
		shape[3] = (crop_w) ? crop_w : datum_width;
		return shape;
	}


	template <typename Dtype>
	void LandmarkDataTransformer<Dtype>::InitRand() {
		const bool needs_rand = param_.mirror() ||
			(phase_ == TRAIN && param_.crop_size());
		if (needs_rand) {
			const unsigned int rng_seed = caffe_rng_rand();
			rng_.reset(new Caffe::RNG(rng_seed));
		}
		else {
			rng_.reset();
		}
	}

	template <typename Dtype>
	int LandmarkDataTransformer<Dtype>::Rand(int n) {
		CHECK(rng_);
		CHECK_GT(n, 0);
		caffe::rng_t* rng =
			static_cast<caffe::rng_t*>(rng_->generator());
		return ((*rng)() % n);
	}

	INSTANTIATE_CLASS(LandmarkDataTransformer);

}  // namespace caffe
