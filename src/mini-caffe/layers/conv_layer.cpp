#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

void ConvolutionLayer::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

void ConvolutionLayer::Forward_cpu(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {

  const real_t* weight = this->blobs_[0]->cpu_data();
  this->weight_num = this->blobs_[0]->num();
  this->weight_ch = this->blobs_[0]->channels();
  this->weight_h = this->blobs_[0]->height();
  this->weight_w = this->blobs_[0]->width();
  for (int i = 0; i < bottom.size(); ++i) {
    const real_t* bottom_data = bottom[i]->cpu_data();
    real_t* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const real_t* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(ConvolutionLayer);
#endif

}  // namespace caffe
