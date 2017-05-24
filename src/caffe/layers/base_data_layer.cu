#include <vector>
#include <opencv/cv.hpp>
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

  Dtype *vis_data_0 = top[0]->mutable_cpu_data();

  vector<cv::Mat > vis_channels_0;
  for (int i = 0; i < 4; i++)
  {
	  cv::Mat vis_transformed_data(368, 368, CV_32FC1, vis_data_0);
	  cv::Mat tmp;
	  vis_transformed_data.copyTo(tmp);
	  tmp = tmp * 256 + 128.0;
	  tmp.convertTo(tmp, CV_8U);
	  vis_channels_0.push_back(tmp);
	  vis_data_0 += 368 * 368;
  }
  
  int stride =8;


  Dtype *vis_data = top[1]->mutable_cpu_data();
  vector<cv::Mat > vis_channels;
  for (int i = 0; i < 62; i++)
  {
	  cv::Mat vis_transformed_data(368 / stride, 368 / stride, CV_32FC1, vis_data);

	  vis_channels.push_back(vis_transformed_data);
	  vis_data += 368 * 368 / (stride* stride);
	  cv::Mat tmp;
	  vis_transformed_data.copyTo(tmp);
	  tmp = tmp * 256;

	  tmp.convertTo(tmp, CV_8U);
	  char zz[256];
	  sprintf_s(zz, "F:/CoreLib/caffe-windows/Build/x64/Debug/%04d.jpg", i);
	  std::string str(zz);
	  imwrite(str, tmp);
  }



  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
