#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <string>

#include "caffe/common.hpp"
#include "caffe/layers/cpm_hand_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

	template <typename Dtype>
	CPMHandDataLayer<Dtype>::CPMHandDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param),
		reader_(param),
		cpm_transform_param_(param.cpm_hand_transform_param()){
	}

	template <typename Dtype>
	CPMHandDataLayer<Dtype>::~CPMHandDataLayer() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void CPMHandDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		cpm_data_transformer_.reset(
			new CPMHandDataTransformer<Dtype>(cpm_transform_param_, this->phase_));
		cpm_data_transformer_->InitRand();


		// Read a data point, and use it to initialize the top blob.
		Datum& datum = *(reader_.full().peek());
		LOG(INFO) << datum.height() << " " << datum.width() << " " << datum.channels();

		bool force_color = this->layer_param_.data_param().force_encoded_color();
		if ((force_color && DecodeDatum(&datum, true)) ||
			DecodeDatumNative(&datum)) {
			LOG(INFO) << "Decoding Datum";
		}

		// image
		const int crop_size = this->layer_param_.cpm_hand_transform_param().crop_size();
		const int batch_size = this->layer_param_.data_param().batch_size();
		if (crop_size > 0) {
			// top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
			// for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			//   this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
			// }
			// //this->transformed_data_.Reshape(1, 4, crop_size, crop_size);
			// this->transformed_data_.Reshape(1, 6, crop_size, crop_size);
		}
		else {
			const int height = this->phase_ != TRAIN ? datum.height() :
				this->layer_param_.cpm_hand_transform_param().crop_size_y();
			const int width = this->phase_ != TRAIN ? datum.width() :
				this->layer_param_.cpm_hand_transform_param().crop_size_x();
			LOG(INFO) << "PREFETCH_COUNT is " << this->PREFETCH_COUNT;
			top[0]->Reshape(batch_size, datum.channels(), height, width);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), height, width);
			}
			//this->transformed_data_.Reshape(1, 4, height, width);
			this->transformed_data_.Reshape(1, datum.channels(), height, width);
		}
		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();

		// label
		if (this->output_labels_) {
			const int stride = this->layer_param_.cpm_hand_transform_param().stride();
			const int height = this->phase_ != TRAIN ? datum.height() :
				this->layer_param_.cpm_hand_transform_param().crop_size_y();
			const int width = this->phase_ != TRAIN ? datum.width() :
				this->layer_param_.cpm_hand_transform_param().crop_size_x();

			int num_parts = this->layer_param_.cpm_hand_transform_param().num_parts();
			top[1]->Reshape(batch_size, 2 * (num_parts + 1), height / stride, width / stride);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].label_.Reshape(batch_size, 2 * (num_parts + 1), height / stride, width / stride);
			}
			this->transformed_label_.Reshape(1, 2 * (num_parts + 1), height / stride, width / stride);
		}
	}

	// This function is called on prefetch thread
	template<typename Dtype>
	void CPMHandDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

		CPUTimer batch_timer;
		batch_timer.Start();
		double deque_time = 0;
		double decod_time = 0;
		double trans_time = 0;
		static int cnt = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());

		// Reshape on single input batches for inputs of varying dimension.
		const int batch_size = this->layer_param_.data_param().batch_size();
		const int crop_size = this->layer_param_.cpm_hand_transform_param().crop_size();
		bool force_color = this->layer_param_.data_param().force_encoded_color();
		if (batch_size == 1 && crop_size == 0) {
			Datum& datum = *(reader_.full().peek());
			if (datum.encoded()) {
				if (force_color) {
					DecodeDatum(&datum, true);
				}
				else {
					DecodeDatumNative(&datum);
				}
			}
			batch->data_.Reshape(1, datum.channels(),
				datum.height(), datum.width());
			this->transformed_data_.Reshape(1, datum.channels(),
				datum.height(), datum.width());
		}

		Dtype* top_data = batch->data_.mutable_cpu_data();
		Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

		if (this->output_labels_) {
			top_label = batch->label_.mutable_cpu_data();
		}
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			Datum& datum = *(reader_.full().pop("Waiting for data"));
			deque_time += timer.MicroSeconds();

			timer.Start();
			cv::Mat cv_img;
			if (datum.encoded()) {
				if (force_color) {
					cv_img = DecodeDatumToCVMat(datum, true);
				}
				else {
					cv_img = DecodeDatumToCVMatNative(datum);
				}
				if (cv_img.channels() != this->transformed_data_.channels()) {
					LOG(WARNING) << "Your dataset contains encoded images with mixed "
						<< "channel sizes. Consider adding a 'force_color' flag to the "
						<< "model definition, or rebuild your dataset using "
						<< "convert_imageset.";
				}
			}
			decod_time += timer.MicroSeconds();

			// Apply data transformations (mirror, scale, crop...)
			timer.Start();
			const int offset_data = batch->data_.offset(item_id);
			const int offset_label = batch->label_.offset(item_id);
			this->transformed_data_.set_cpu_data(top_data + offset_data);
			this->transformed_label_.set_cpu_data(top_label + offset_label);
			if (datum.encoded()) {
				this->cpm_data_transformer_->Transform(cv_img, &(this->transformed_data_));
			}
			else {
				this->cpm_data_transformer_->Transform_nv(datum,
					&(this->transformed_data_),
					&(this->transformed_label_), cnt);
				++cnt;
			}

			// for debug
			/*	const int height = datum.height();

			const int width = datum.width();
			const int channel = datum.channels();
			*/
			Dtype *vis_data_0 = this->transformed_data_.mutable_cpu_data();

			vector<cv::Mat > vis_channels_0;
			for (int i = 0; i < 4; i++)
			{
				cv::Mat vis_transformed_data(368, 368, CV_32FC1, vis_data_0);
				vis_transformed_data = vis_transformed_data * 256 + 128.0;
				vis_transformed_data.convertTo(vis_transformed_data, CV_8U);
				vis_channels_0.push_back(vis_transformed_data);
				vis_data_0 += 368 * 368;
			}
			const int height = datum.height();

			const int width = datum.width();

			const int stride = this->layer_param_.cpm_hand_transform_param().stride();
			int num_parts = this->layer_param_.cpm_hand_transform_param().num_parts();

			Dtype *vis_data = this->transformed_label_.mutable_cpu_data();
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

			//-----

			if (this->output_labels_) {
				top_label[item_id] = datum.label();
			}
			trans_time += timer.MicroSeconds();

			reader_.free().push(const_cast<Datum*>(&datum));
		}
		batch_timer.Stop();

		VLOG(2) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		VLOG(2) << "  Dequeue time: " << deque_time / 1000 << " ms.";
		VLOG(2) << "   Decode time: " << decod_time / 1000 << " ms.";
		VLOG(2) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(CPMHandDataLayer);
	REGISTER_LAYER_CLASS(CPMHandData);

}  // namespace caffe