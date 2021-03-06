import caffe
from caffe import layers as L
from caffe import params as P

import  LayerCreator


class HandMobileNet(object):
    def __init__(self, lmdb_train):
        self.train_data = lmdb_train
        self.n = caffe.NetSpec()
        self.LC = LayerCreator.LayerCreator(self.n)

    def layers_proto(self, phase='TRAIN'):
        n = self.n
        input_w = 256
        input_h = 256
        batch_size = 12
        stride = 8
        np_in_lmdb = 21
        num_paf = (np_in_lmdb - 1) * 2
        stages = 2
        dim = 128

        data_source = self.train_data

        label_name = ['label_vec', 'label_heat', 'label_slice']
        label_name_heatmap = label_name[1]

        transform_param = dict(stride=stride, crop_size_x=input_w, crop_size_y=input_h,
                                    scale_prob=1, scale_min=0.5, scale_max=1.1,
                                 max_rotate_degree=40, center_perterb_max=20, hand_padding=1.2,
                                 do_aug=False, np_in_lmdb=np_in_lmdb, num_parts=num_paf + np_in_lmdb + 1)
        
       
        num_parts = transform_param['num_parts']

        if phase == 'TRAIN':
            n.data, n.tops['label'] = L.CPMHandData(data_param=dict(backend=1, source=data_source, batch_size=batch_size), 
                                                    cpm_hand_transform_param=transform_param, ntop=2)
            n.tops[label_name[0]], n.tops[label_name[1]] = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[40]), ntop=2)
            n.notuse0 =  L.Silence(n.tops[label_name[0]])
           
        else:
            input = "data"
            dim1 = 1
            dim2 = 4
            dim3 = input_h
            dim4 = input_w
            # make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
            # we will later have to remove this layer from the serialization string, since this is just a placeholder
            n.data = L.Layer()

        
        n.conv4_4_CPM, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
        n.notuse1 =  L.Silence(n.center_map)
        l = n.conv4_4_CPM
        LC = self.LC
        for s in range(0, 2):
            l1 = l
            l2 = l
            for k in range(0, 3):
                l1 = LC.DepthWithBlock(l1, str(k+1) + "_stage1_L1", dim)
                l2 = LC.DepthWithBlock(l2, str(k+1) + "_stage1_L2", dim)
            
            l1 = LC.ConvReLU(l1,512)
            l2 = LC.ConvReLU(l2,512)
            l1 = LC.Conv(l1, num_paf)
            l2 = LC.Conv(l2, np_in_lmdb + 1)

            if phase == 'TRAIN':
                loss1 = LC.EuclideanLoss(l1, n.label_vec)
                loss2 = LC.EuclideanLoss(l2, n.label_heat)
            if s < stages - 1:
                l0 = LC.Concat(l1,l2)
                l = LC.Concat(l,l0)
        


        return n.to_proto()



