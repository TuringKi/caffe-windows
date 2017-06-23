#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void SplitLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

}  // namespace caffe
