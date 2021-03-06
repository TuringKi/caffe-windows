#include <vector>

#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void InnerProductLayer::LayerSetUp(const vector<Blob*>& bottom,
                                   const vector<Blob*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob(weight_shape));
    // fill the weights
    shared_ptr<Filler> weight_filler(GetFiller(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob(bias_shape));
      shared_ptr<Filler> bias_filler(GetFiller(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
}

void InnerProductLayer::Reshape(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, static_cast<real_t>(1), bias_multiplier_.mutable_cpu_data());
  }
}

void InnerProductLayer::Forward_cpu(const vector<Blob*>& bottom,
                                    const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const real_t* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
    M_, N_, K_, static_cast<real_t>(1),
    bottom_data, weight, static_cast<real_t>(0), top_data);
  if (bias_term_) {
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, static_cast<real_t>(1),
      bias_multiplier_.cpu_data(),
      this->blobs_[1]->cpu_data(), static_cast<real_t>(1), top_data);
  }
}

#ifndef USE_CUDA
STUB_GPU(InnerProductLayer);
#endif

REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
