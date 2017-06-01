import caffe
from caffe import layers as L
from caffe import params as P

import  LayerCreatorWithoutBN as LayerCreator


def conv_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=1):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='gaussian', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    relu = L.ReLU(conv, in_place=True)
    return conv, relu


class HandHourglassNet(object):
    def __init__(self, lmdb_train):
        self.train_data = lmdb_train
        self.n = caffe.NetSpec()
        self.LC = LayerCreator.LayerCreator(self.n)

    def linear(self, input, num_output):
        output = self.LC.ConvReLU(input, num_output)
        return output
    def hourglass(self, level, dim, n_modules, input):
    
        # upper branch:
        up1 = input
        for i in range(0, n_modules):
  
            up1 = self.LC.Residual(up1, dim, dim)
        
        # lower branch:
        low1 = self.LC.Pool(input)
        
        for i in range(0, n_modules):
            low1 = self.LC.Residual(low1, dim, dim)
        if level > 1:
            low2 = self.hourglass(level - 1, dim, n_modules, low1)
            # print low2
        else:
            low2 = low1
            for i in range(0, n_modules):
                low2 = self.LC.Residual(low2, dim, dim)
        
        low3 = low2
        for i in range(0, n_modules):
            low3 = self.LC.Residual(low3, dim, dim)
        up2 = self.LC.Upsamping(low3, dim)

        up = self.LC.Add(up1, up2)
        return up

    def layers_proto(self, phase='TRAIN'):
        n = self.n
        input_w = 256
        input_h = 256
        batch_size = 6
        stride = 4
        np_in_lmdb = 21
        num_paf = (np_in_lmdb - 1) * 2
        stacks = 6
        n_modules = 1 #number of residual blocks for each feature extraction block
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

        
        n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
        n.notuse1 =  L.Silence(n.center_map)
        conv1 = \
        self.LC.ConvReLU(n.image, num_output=64, kernel_size=7, stride=2, pad=3)  # 64 x input_w/2 x input_h/2
        # n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64 x input_w/4 x input_h/4
        
        r1 = self.LC.Residual(conv1, 64, 128)
        pool = self.LC.Pool(r1)  # 64 x input_w/4 x input_h/4
        r2 = self.LC.Residual( pool,  128, 128)
        
        r3 = self.LC.Residual( r2,  128, dim)

        inter = r3


        for j in range(0, stacks):
            print 'Creating stack %d...' % (j)
            hg = self.hourglass(4,dim, n_modules, inter)
            
            ll = hg
            # Residual layers at output resolution
            for i in range(0, n_modules):
                ll = self.LC.Residual(ll, dim, dim)
            
            ll = self.linear(ll,dim)

            # prediction output:
            pred = self.LC.Conv(ll,np_in_lmdb + 1)
            
            if phase == 'TRAIN':
                # add loss layer:
                loss = self.LC.EuclideanLoss(pred, n.label_heat)
            
            if j < (stacks - 1):
                #print j
                ll_ = self.LC.Conv(ll, dim)
                pred_ = self.LC.Conv(pred, dim)
                inter_ = self.LC.Add(ll_, pred_)
                inter = self.LC.Add(inter_, inter)

        return n.to_proto()
    
    def vgg_proto(self, phase='TRAIN'):
        n = self.n
        input_w = 256
        input_h = 256
        batch_size = 6
        stride = 4
        np_in_lmdb = 21
        num_paf = (np_in_lmdb - 1) * 2
        stacks = 6
        n_modules = 1 #number of residual blocks for each feature extraction block
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

        
        n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
        n.notuse1 =  L.Silence(n.center_map)

        n.conv1_1, n.relu1_1 = conv_relu(n.image, num_output=64)
        n.conv1_2, n.relu1_2 = conv_relu(n.conv1_1, num_output=64)
        n.pool1 = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, num_output=128)
        n.conv2_2, n.relu2_2 = conv_relu(n.conv2_1, num_output=128)
        n.pool2 = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, num_output=256)
        n.conv3_2, n.relu3_2 = conv_relu(n.conv3_1, num_output=256)
        n.conv3_3, n.relu3_3 = conv_relu(n.conv3_2, num_output=256)
        n.conv3_4, n.relu3_4 = conv_relu(n.conv3_3, num_output=256)

    
        n.conv4_5, n.relu4_5 = conv_relu(n.conv3_4, num_output=dim)

        vgg_out = n.conv4_5
        inter = vgg_out


        for j in range(0, stacks):
            print 'Creating stack %d...' % (j)
            hg = self.hourglass(4,dim, n_modules, inter)
            
            ll = hg
            # Residual layers at output resolution
            for i in range(0, n_modules):
                ll = self.LC.Residual(ll, dim, dim)
            
            ll = self.linear(ll,dim)

            # prediction output:
            pred = self.LC.Conv(ll,np_in_lmdb + 1)
            
            if phase == 'TRAIN':
                # add loss layer:
                loss = self.LC.EuclideanLoss(pred, n.label_heat)
            
            if j < (stacks - 1):
                #print j
                ll_ = self.LC.Conv(ll, dim)
                pred_ = self.LC.Conv(pred, dim)
                inter_ = self.LC.Add(ll_, pred_)
                inter = self.LC.Add(inter_, inter)

        return n.to_proto()


        