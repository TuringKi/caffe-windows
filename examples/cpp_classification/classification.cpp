#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace caffe;

int main(int argc, char** argv)
{
	Net<float> templ_feature_net("F:\\FaceRecognition\\Collections\\MTCNN_face_detection_alignment-master\\code\\codes\\MTCNNv2\\model\\det1-memory.prototxt", TEST);
	templ_feature_net.CopyTrainedLayersFromBinaryProto("F:\\FaceRecognition\\Collections\\MTCNN_face_detection_alignment-master\\code\\codes\\MTCNNv2\\model\\det1.caffemodel");

	return 0;
}