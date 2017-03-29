#include <vector>
#include "caffe/util/im2col.hpp"
#include "caffe/layers/xcorr_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
	Blob<Dtype> *bottom_image, *bottom_target;
	if (bottom[0]->count(1) > bottom[1]->count(1)) {
		bottom_image = bottom[0]; //第一个输入应该是被卷积的图像
		bottom_target = bottom[1]; //第二个输入为卷积核，也就是模板
	}
	else {
		bottom_image = bottom[1];
		bottom_target = bottom[0];

	}

	const Dtype* weights = bottom_target->gpu_data();
	//const Dtype* weights_cpu = bottom_target->cpu_data();

	const Dtype* bottom_data = bottom_image->gpu_data();

	Dtype* top_data = top[0]->mutable_gpu_data();

	for (int n = 0; n < this->image_num_; ++n) {

		const Dtype* input = bottom_data + n * this->conv_in_spatial_dim_;
		const Dtype* weight = weights + n * this->kernel_dim_;
		Dtype* output = top_data + n * this->conv_out_spatial_dim_;

		const int* pads = pad_.gpu_data();
		const int* strides = stride_.gpu_data();
		im2col_gpu<Dtype>(input, this->channels_,
			this->height_, this->width_, this->filter_height_, this->filter_width_,
			pads[0], pads[1],
			strides[0], strides[1], 1, 1,
			col_buffer_.mutable_gpu_data());

		const Dtype* col_buff = col_buffer_.gpu_data();

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
			conv_out_spatial_dim_, kernel_dim_,
			alpha_, weight, col_buff,
			Dtype(0.), output);


		caffe_gpu_add_scalar(conv_out_spatial_dim_, (Dtype)beta_, output);

	}
	

}




template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	Blob<Dtype> *bottom_image, *bottom_target;
	if (bottom[0]->count(1) > bottom[1]->count( 1)) {
		bottom_image = bottom[0]; //第一个输入应该是被卷积的图像
		bottom_target = bottom[1]; //第二个输入为卷积核，也就是模板
	}
	else {
		bottom_image = bottom[1];
		bottom_target = bottom[0];

	}

	const Dtype* weight = bottom_target->gpu_data();
	Dtype* weight_diff = bottom_target->mutable_gpu_diff();

	const Dtype* top_diff = top[0]->gpu_diff();

	const Dtype* image_data = bottom_image->gpu_data();
	Dtype* image_diff = bottom_image->mutable_gpu_diff();

	for (int n = 0; n < this->image_num_; ++n) {
		
		const Dtype* input = image_data + n * conv_in_spatial_dim_;
		const Dtype* output = top_diff + n * conv_out_spatial_dim_;
		Dtype* weights = weight_diff + n * this->kernel_dim_;
		
		// gradient w.r.t. weight. Note that we will accumulate diffs.

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1,
			kernel_dim_, conv_out_spatial_dim_,
			alpha_, output, input,
			(Dtype)1., weight_diff + n * this->kernel_dim_);


		// gradient w.r.t. bottom data, if necessary.
	
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
			conv_out_spatial_dim_, 1,
			alpha_, weight + n * this->kernel_dim_, output,
			(Dtype)0., image_diff + n * this->conv_in_spatial_dim_);
		
	}

 
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossCorrelationLayer);

}  // namespace caffe
