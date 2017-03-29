#ifndef CAFFE_XCORR_LAYER_HPP_
#define CAFFE_XCORR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 *  通过将target的输出视为卷积核，instance的
 *  输出视为被卷积对象，
 *  实现Siamese-fc中的target和instance的相似度计算。
 *   
 */
template <typename Dtype>
class CrossCorrelationLayer : public Layer<Dtype> {
 public:
  
	 explicit CrossCorrelationLayer(const LayerParameter& param)
		 : Layer<Dtype>(param) {}
 
	 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
	 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "XCorr"; }
  
  virtual inline int MinBottomBlobs() const { return 1; }
  
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  // "滤波器"——target的参数
  int image_num_, filter_num_;
  int channels_;
  int height_, width_;
  int filter_height_, filter_width_;



  // Cross Correlation操作的参数
  int channel_axis_;
  int num_spatial_axes_;
  Blob<int> pad_;
  Blob<int> stride_;
  int group_;
  vector<int> output_shape_;
  int out_spatial_dim_;
  
  Dtype alpha_;
  Dtype beta_;


  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int conv_in_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  vector<int> col_buffer_shape_;

};

}  // namespace caffe

#endif  // CAFFE_XCORR_LAYER_HPP_
