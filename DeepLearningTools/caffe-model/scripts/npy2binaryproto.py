import caffe

import numpy as np


np.set_printoptions(threshold='nan')


MEAN_FILE = 'VGG_mean.binaryproto'

NPY_FILE = 'ilsvrc_2012_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data_mean = []
with open(NPY_FILE,'rb') as f:
     data_mean = np.load(f)
blob.shape.dim.extend(data_mean.shape)
blob.num=1
blob.channels,blob.height,blob.width=data_mean.shape
blob.data.extend(data_mean.astype(float).flat)
#blob=caffe.io.array_to_blobproto(data_mean)
print blob.num
print blob.channels
print blob.height
print blob.width
binaryproto_file = open('mean.binaryproto', 'wb' )
binaryproto_file.write(blob.SerializeToString())
binaryproto_file.close()