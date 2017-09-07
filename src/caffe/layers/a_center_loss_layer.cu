#include <vector>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/a_center_loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void Weight_norm_gpu(int nthreads, const int K_,
		Dtype* weight) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype sum_sqaure = 0.;
			for (int i = 0; i < K_; i++) {
				sum_sqaure += weight[index * K_ + i] * weight[index * K_ + i];
			}
			sum_sqaure = sqrt(sum_sqaure);
			for (int i = 0; i < K_; i++) {
				weight[index * K_ + i] = weight[index * K_ + i] / sum_sqaure;
			}
		}
	}

	template <typename Dtype>
	__global__ void Compute_bottom_norm_gpu(int nthreads, const int K_,
		const Dtype* bottom, Dtype* x_norm) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype sum_sqaure = 0.;
			for (int i = 0; i < K_; i++) {
				sum_sqaure += bottom[index * K_ + i] * bottom[index * K_ + i];
			}
			x_norm[index] = sqrt(sum_sqaure);
		}
	}

	template <typename Dtype>
	__global__ void Compute_cos_theta_gpu(int nthreads, const int N_,
		const Dtype* x_norm, Dtype* cos_theta) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int i = index / N_;
			cos_theta[index] = cos_theta[index] / x_norm[i];
		}
	}

	template <typename Dtype>
	__global__ void Compute_sign_1_gpu(int nthreads, const Dtype* cos_theta, Dtype* sign_1) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			sign_1[index] = abs(cos_theta[index]) - (Dtype)0.5;
		}
	}

	template <typename Dtype>
	__global__ void Compute_sign_2_gpu(int nthreads, const Dtype* sign_0,
		const Dtype* sign_1, Dtype* sign_2) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			sign_2[index] = sign_0[index] * ((Dtype)1. + sign_1[index]) - (Dtype)2.;
		}
	}

	template <typename Dtype>
	__global__ void Compute_sign_3_gpu(int nthreads, const Dtype* sign_0,
		const Dtype* cos_theta_quadratic, Dtype* sign_3) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			sign_3[index] = sign_0[index] * ((Dtype)2. * cos_theta_quadratic[index] - (Dtype)1.);
		}
	}

	template <typename Dtype>
	__global__ void Compute_sign_4_gpu(int nthreads, const Dtype* sign_0,
		const Dtype* sign_3, Dtype* sign_4) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			sign_4[index] = (Dtype)2. * sign_0[index] + sign_3[index] - (Dtype)3.;
		}
	}

	template <typename Dtype>
	__global__ void Margin_single_forward_gpu(int nthreads, const int N_, Dtype lambda,
		const Dtype* label, const Dtype* x_norm, const Dtype* sign_0,
		const Dtype* cos_theta_quadratic, Dtype* top, int base_) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// the label[i]_th top_data
			const int i = index / N_;
			top[index] /= x_norm[i];

		}
	}

	
	template <typename Dtype>
	__global__ void Margin_bottom_single_backward_gpu(int nthreads, const int N_, const int K_, Dtype lambda,
		const Dtype* bottom, const Dtype* weight, const Dtype* top_diff, const Dtype* label,
		const Dtype* x_norm, Dtype* bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int i = index / K_;
			const int j = index % K_;
			bottom_diff[index] = (Dtype)0.;
			const int label_value = static_cast<int>(label[i]);
			for (int n = 0; n < N_; n++)
				bottom_diff[index] += top_diff[i * N_ + n] * weight[n * K_ + j] / x_norm[i];
		}
	}


	template <typename Dtype>
	__global__ void Compute_centerloss_gpu(int nthreads, const int N, const int M,
		const Dtype* label, const Dtype* sim, Dtype* loss) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			for (int n = 0; n < M; n++){
				const int label_value = static_cast<int>(label[n]);
				*loss += sim[M*label_value];
			}
			
			
		}
	}


	
	template <typename Dtype>
	void ACenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		iter_ += (Dtype)1.;
		Dtype base_ = this->layer_param_.margin_inner_product_param().base();
		Dtype gamma_ = this->layer_param_.margin_inner_product_param().gamma();
		Dtype power_ = this->layer_param_.margin_inner_product_param().power();
		Dtype lambda_min_ = this->layer_param_.margin_inner_product_param().lambda_min();
		lambda_ = base_ * powf(((Dtype)1. + gamma_ * iter_), -power_);
		lambda_ = max(lambda_, lambda_min_);
		top[1]->mutable_cpu_data()[0] = lambda_;

		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* label = bottom[1]->gpu_data();
		
		/************************* normalize weight *************************/
		int nthreads = N_;
		Weight_norm_gpu<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS >> >(nthreads, K_,
			this->blobs_[0]->mutable_gpu_data());
	
		/************************* common variables *************************/
		// x_norm_ = |x|
		nthreads = M_;
		Compute_bottom_norm_gpu<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS >> >(nthreads, K_, bottom_data,
			x_norm_.mutable_gpu_data());
		

		nthreads = M_ * N_;
		// cos_theta = x'w / |x|
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight, (Dtype)0., cos_theta_.mutable_gpu_data());
		Compute_cos_theta_gpu<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS >> >(nthreads, N_,
			x_norm_.gpu_data(), cos_theta_.mutable_gpu_data());

		caffe_gpu_scal<Dtype>(cos_theta_.count(), (Dtype)0.5, cos_theta_.mutable_gpu_data());
		caffe_gpu_add_scalar<Dtype>(cos_theta_.count(), (Dtype)0.5, cos_theta_.mutable_gpu_data());



		auto tmp = cos_theta_.cpu_data();
		auto tmp2 = bottom[1]->cpu_data();

		nthreads = M_ * K_;
		Dtype loss = 0;
		//Compute_centerloss_gpu<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
		//	CAFFE_CUDA_NUM_THREADS >> >(nthreads, K_, bottom[1]->gpu_data(),
		//	cos_theta_.mutable_gpu_data(), &loss);

	//	top[0]->mutable_gpu_data()[0] = loss;

	}

	template <typename Dtype>
	void ACenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label = bottom[1]->gpu_data();
		const Dtype* weight = this->blobs_[0]->gpu_data();
		const Dtype* x_norm = x_norm_.gpu_data();

	}

	INSTANTIATE_LAYER_GPU_FUNCS(ACenterLossLayer);

}  // namespace caffe
