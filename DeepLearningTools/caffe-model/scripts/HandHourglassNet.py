import caffe
from caffe import layers as L
from caffe import params as P

import  LayerCreator





class HandHourglassNet(object):
    def __init__(self, lmdb_train):
        self.train_data = lmdb_train
        self.n = caffe.NetSpec()
        self.LC = LayerCreator.LayerCreator(self.n)

    
    def hourglass(self, level, dim, res_idx, n_modules, input):
    
        # upper branch:
        up1 = input
        for i in range(1, n_modules):
            res_idx += res_idx
            up1 = self.LC.Residual(up1, dim, dim)
        
        # lower branch:
        res_idx += res_idx
        low1 = self.LC.Pool(input)
        
        for i in range(1, n_modules):
            res_idx += res_idx
            low1 = self.LC.Residual(low1, dim, dim)
            


    def layers_proto(self, phase='TRAIN'):
        n = self.n
        input_w = 256
        input_h = 256
        batch_size = 10
        stride = 4
        data_source = self.train_data

        label_name = ['label_vec', 'label_heat', 'label_slice']
        label_name_heatmap = label_name[1]

        transform_param = dict(stride=stride, crop_size_x=input_w, crop_size_y=input_h,
                                    scale_prob=1, scale_min=0.5, scale_max=1.1,
                                 max_rotate_degree=40, center_perterb_max=20, hand_padding=1.2,
                                 do_aug=True, np_in_lmdb=21, num_parts=62)
        
       
        num_parts = transform_param['num_parts']

        if phase == 'TRAIN':
            n.data, n.tops['label'] = L.CPMHandData(data_param=dict(backend=1, source=data_source, batch_size=batch_size), 
                                                    cpm_hand_transform_param=transform_param, ntop=2)
            n.tops[label_name[0]], n.tops[label_name[1]] = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[40]), ntop=2)
        else:
            input = "data"
            dim1 = 1
            dim2 = 4
            dim3 = input_h
            dim4 = input_w
            # make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
            # we will later have to remove this layer from the serialization string, since this is just a placeholder
            n.data = L.Layer()

        
        n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = \
        LayerCreator.conv_bn_scale_relu(n.image, num_output=64, kernel_size=7, stride=2, pad=3)  # 64 x input_w/2 x input_h/2
        # n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64 x input_w/4 x input_h/4
        
        r1 = self.LC.Residual(n.conv1, 64, 128)
        pool = self.LC.Pool(r1)  # 64 x input_w/4 x input_h/4
        r2 = self.LC.Residual( pool,  128, 128)
        
        r3 = self.LC.Residual( r2,  128, 256)

        return n.to_proto()

        