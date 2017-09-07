#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/a_center_loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ACenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "Number of labels must match number of output; "
			<< "DO NOT support multi-label this version."
			<< "e.g., if prediction shape is (M X N), "
			<< "label count (number of labels) must be M, "
			<< "with integer values in {0, 1, ..., N-1}.";

		type_ = this->layer_param_.margin_inner_product_param().type();
		iter_ = this->layer_param_.margin_inner_product_param().iteration();
		lambda_ = (Dtype)0.;

		const int num_output = this->layer_param_.margin_inner_product_param().num_output();
		N_ = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.margin_inner_product_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		K_ = bottom[0]->count(axis);
		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			this->blobs_.resize(1);
			// Intialize the weight
			vector<int> weight_shape(2);
			weight_shape[0] = N_;
			weight_shape[1] = K_;
			this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
			// fill the weights
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
				this->layer_param_.margin_inner_product_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
		}  // parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void ACenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Figure out the dimensions
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.margin_inner_product_param().axis());
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

		// if needed, reshape top[1] to output lambda
		if (top.size() == 2) {
			vector<int> lambda_shape(1, 1);
			top[1]->Reshape(lambda_shape);
		}

		// common temp variables
		vector<int> shape_1_X_M(1, M_);
		x_norm_.Reshape(shape_1_X_M);
		sign_0_.Reshape(top_shape);
		cos_theta_.Reshape(top_shape);

		// optional temp variables
		switch (type_) {
		case MarginInnerProductParameter_MarginType_SINGLE:
			break;
		case MarginInnerProductParameter_MarginType_DOUBLE:
			cos_theta_quadratic_.Reshape(top_shape);
			break;
		case MarginInnerProductParameter_MarginType_TRIPLE:
			cos_theta_quadratic_.Reshape(top_shape);
			cos_theta_cubic_.Reshape(top_shape);
			sign_1_.Reshape(top_shape);
			sign_2_.Reshape(top_shape);
			break;
		case MarginInnerProductParameter_MarginType_QUADRUPLE:
			cos_theta_quadratic_.Reshape(top_shape);
			cos_theta_cubic_.Reshape(top_shape);
			cos_theta_quartic_.Reshape(top_shape);
			sign_3_.Reshape(top_shape);
			sign_4_.Reshape(top_shape);
			break;
		default:
			LOG(FATAL) << "Unknown margin type.";
		}
	}

	template <typename Dtype>
	void ACenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
	}

	template <typename Dtype>
	void ACenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(ACenterLossLayer);
#endif

	INSTANTIATE_CLASS(ACenterLossLayer);
	REGISTER_LAYER_CLASS(ACenterLoss);

}  // namespace caffe
