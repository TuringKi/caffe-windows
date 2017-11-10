#ifndef CAFFE_LANDMARK_DATA_TRANSFORMER_HPP
#define CAFFE_LANDMARK_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Applies common transformations to the input data, such as
	* scaling, mirroring, substracting the image mean...
	*/
	template <typename Dtype>
	class LandmarkDataTransformer {
	public:
		explicit LandmarkDataTransformer(const TransformationParameter& param, Phase phase);
		virtual ~LandmarkDataTransformer() {}


		void InitRand();


		void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);


		vector<int> InferBlobShape(const Datum& datum);

	protected:
		
		bool FaceAlignment(cv::Mat input_image,
			const std::vector<cv::Point2f> &face_points, cv::Size size, cv::Mat &warp_dst);
	


		virtual int Rand(int n);

		void Transform(const Datum& datum, Dtype* transformed_data);
		
			
		// Tranformation parameters
		TransformationParameter param_;


		shared_ptr<Caffe::RNG> rng_;
		Phase phase_;
		Blob<Dtype> data_mean_;
		vector<Dtype> mean_values_;
	};

	

}  // namespace caffe

#endif  // CAFFE_LANDMARK_DATA_TRANSFORMER_HPP
