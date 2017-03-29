#include <vector>
#include "caffe/util/im2col.hpp"
#include "caffe/layers/xcorr_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define CUDNN_STREAMS_PER_GROUP 3

template <typename Dtype>
void CrossCorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
	CHECK_EQ(bottom.size(), 2) << this->type() << " Layer must have two inputs.";// 输入必须为两个Blob！

	Blob<Dtype> *bottom_image, *bottom_target;
	if (bottom[0]->count(1) > bottom[1]->count(1)) {
		bottom_image = bottom[0]; //第一个输入应该是被卷积的图像
		bottom_target = bottom[1]; //第二个输入为卷积核，也就是模板
	}
	else {
		bottom_image = bottom[1];
		bottom_target = bottom[0];

	}

	int image_N = bottom_image->num();
	int image_C = bottom_image->channels();
	int target_N = bottom_target->num();
	int target_C = bottom_target->channels();
	int image_H = bottom_image->height();
	int image_W = bottom_image->width();
	int target_H = bottom_target->height();
	int target_W = bottom_target->width();



	//检查 N , C, 
	CHECK_EQ(image_N, target_N) 
		<< this->type() << " Layer: the two inputs must have the same nums.";
	CHECK_EQ(image_C, target_C) 
		<< this->type() << "  Layer: the two inputs must have the same channels.";
	

	channels_ = image_C;
	height_ = image_H;
	width_ = image_W;
	filter_height_ = target_H;
	filter_width_ = target_W;
	image_num_ = image_N;
	filter_num_ = target_N;

	XCorrParameter xcorr_param = this->layer_param_.xcorr_param();
	
	alpha_ = xcorr_param.alpha();
	beta_ = xcorr_param.beta();
	
	channel_axis_ = bottom[0]->CanonicalAxisIndex(xcorr_param.axis());

	const int first_spatial_axis = channel_axis_ + 1;
	num_spatial_axes_ = bottom[0]->num_axes() - first_spatial_axis;
	
	CHECK_GE(num_spatial_axes_, 0);
	vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));

	// Setup stride dimensions (stride_).
	stride_.Reshape(spatial_dim_blob_shape);
	int* stride_data = stride_.mutable_cpu_data();

	if (xcorr_param.has_stride_h() || xcorr_param.has_stride_w()) {
		CHECK_EQ(num_spatial_axes_, 2)
			<< "stride_h & stride_w can only be used for 2D convolution.";
		CHECK_EQ(0, xcorr_param.stride_size())
			<< "Either stride or stride_h/w should be specified; not both.";
		stride_data[0] = xcorr_param.stride_h();
		stride_data[1] = xcorr_param.stride_w();
	}
	else {
		const int num_stride_dims = xcorr_param.stride_size();
		CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
			num_stride_dims == num_spatial_axes_)
			<< "stride must be specified once, or once per spatial dimension "
			<< "(stride specified " << num_stride_dims << " times; "
			<< num_spatial_axes_ << " spatial dims).";
		const int kDefaultStride = 1;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
				xcorr_param.stride((num_stride_dims == 1) ? 0 : i);
			CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
		}
	}
	
	// Setup pad dimensions (pad_).
	pad_.Reshape(spatial_dim_blob_shape);
	int* pad_data = pad_.mutable_cpu_data();
	if (xcorr_param.has_pad_h() || xcorr_param.has_pad_w()) {
		CHECK_EQ(num_spatial_axes_, 2)
			<< "pad_h & pad_w can only be used for 2D convolution.";
		CHECK_EQ(0, xcorr_param.pad_size())
			<< "Either pad or pad_h/w should be specified; not both.";
		pad_data[0] = xcorr_param.pad_h();
		pad_data[1] = xcorr_param.pad_w();
	}
	else {
		const int num_pad_dims = xcorr_param.pad_size();
		CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
			num_pad_dims == num_spatial_axes_)
			<< "pad must be specified once, or once per spatial dimension "
			<< "(pad specified " << num_pad_dims << " times; "
			<< num_spatial_axes_ << " spatial dims).";
		const int kDefaultPad = 0;
		for (int i = 0; i < num_spatial_axes_; ++i) {
			pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
				xcorr_param.pad((num_pad_dims == 1) ? 0 : i);
		}
	}
	group_ = xcorr_param.group();

	conv_out_channels_ = 1;
	conv_in_channels_ = channels_;

}

template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(bottom.size(), 2) << this->type() << " Layer must have two inputs.";// 输入必须为两个Blob！

	const int first_spatial_axis = channel_axis_ + 1;

	Blob<Dtype> *bottom_image, *bottom_target;
	if (bottom[0]->count( 1) > bottom[1]->count( 1)) {
		bottom_image = bottom[0]; //第一个输入应该是被卷积的图像
		bottom_target = bottom[1]; //第二个输入为卷积核，也就是模板
	}
	else {
		bottom_image = bottom[1];
		bottom_target = bottom[0];

	}
	 

	const std::vector<int> kernel_shape_data = bottom_target->shape();
	const int* stride_data = this->stride_.cpu_data();
	const int* pad_data = this->pad_.cpu_data();
	
	output_shape_.clear();

	output_shape_.push_back(image_num_); // N
	output_shape_.push_back(1);// C
	for (int i = first_spatial_axis; i < 4; ++i) { // H, W
		// i + 1 to skip channel axis
		const int input_dim = bottom_image->shape(i);
		const int kernel_extent =(kernel_shape_data[i] - 1) + 1;
		const int output_dim = (input_dim + 2 * pad_data[i - first_spatial_axis] - kernel_extent)
			/ stride_data[i - first_spatial_axis] + 1;
		output_shape_.push_back(output_dim);
	}

	top[0]->Reshape(output_shape_);
	conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
	kernel_dim_ = bottom_target->count(1);
	conv_in_spatial_dim_ = bottom_image->count(1);


	col_buffer_shape_.clear();
	col_buffer_shape_.push_back(kernel_dim_);
	for (int i = 0; i < num_spatial_axes_; ++i) {
	
		col_buffer_shape_.push_back(output_shape_[i+2]);
		
	}
	col_buffer_.Reshape(col_buffer_shape_);

}



template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 2) << this->type() << " Layer must have two inputs.";// 输入必须为两个Blob！


	Blob<Dtype> *bottom_image, *bottom_target;
	if (bottom[0]->count(1) > bottom[1]->count(1)) {
		bottom_image = bottom[0]; //第一个输入应该是被卷积的图像
		bottom_target = bottom[1]; //第二个输入为卷积核，也就是模板
	}
	else {
		bottom_image = bottom[1];
		bottom_target = bottom[0];

	}

	const Dtype* weights = bottom_target->cpu_data();

	const Dtype* bottom_data = bottom_image->cpu_data();

	Dtype* top_data = top[0]->mutable_cpu_data();

	for (int n = 0; n < this->image_num_; ++n) {

		const Dtype* input = bottom_data + n * this->conv_in_spatial_dim_;
		const Dtype* weight = weights + n * this->kernel_dim_;
		Dtype* output = top_data + n * this->conv_out_spatial_dim_;

		const int * pads = pad_.cpu_data();
		const int * strides = stride_.cpu_data();

		im2col_cpu<Dtype>(input, this->channels_,
			this->height_, this->width_, this->filter_height_, this->filter_width_,
			pads[0], pads[1],
			strides[0], strides[1],1, 1,
			col_buffer_.mutable_cpu_data());
		
		const Dtype* col_buff = col_buffer_.cpu_data();

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1,
			conv_out_spatial_dim_, kernel_dim_,
			alpha_, weight, col_buff,
			Dtype(0.), output);

		caffe_add_scalar(conv_out_spatial_dim_, (Dtype)beta_, output);
	}


}

template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

	const Dtype* weight = bottom_target->cpu_data();
	Dtype* weight_diff = bottom_target->mutable_cpu_diff();

	const Dtype* top_diff = top[0]->cpu_diff();

	const Dtype* image_data = bottom_image->cpu_data();
	Dtype* image_diff = bottom_image->mutable_cpu_diff();

	for (int n = 0; n < this->image_num_; ++n) {

		const Dtype* input = image_data + n * conv_in_spatial_dim_;
		const Dtype* output = top_diff + n * conv_out_spatial_dim_;
		Dtype* weights = weight_diff + n * this->kernel_dim_;

		// gradient w.r.t. weight. Note that we will accumulate diffs.

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1,
			kernel_dim_, conv_out_spatial_dim_,
			alpha_, output, input,
			(Dtype)1., weight_diff + n * this->kernel_dim_);


		// gradient w.r.t. bottom data, if necessary.

		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
			conv_out_spatial_dim_, 1,
			alpha_, weight + n * this->kernel_dim_, output,
			(Dtype)0., image_diff + n * this->conv_in_spatial_dim_);

	}


}

#ifdef CPU_ONLY
STUB_GPU(CrossCorrelationLayer);
#endif

INSTANTIATE_CLASS(CrossCorrelationLayer);
REGISTER_LAYER_CLASS(CrossCorrelation);

}  // namespace caffe
