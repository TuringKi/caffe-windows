import caffe
from caffe import layers as L
from caffe import params as P

import math

def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant', value=0))
    conv_relu = L.PReLU(conv, in_place=True)

    return conv, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier'),
                         bias_filler=dict(type='constant', value=0))

    return conv

def eltwize_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.PReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64, num_output=256):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch2a, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=64, kernel_size=1)  # base_output x n x n
    branch2b, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=64, kernel_size=3, pad=1)  # base_output x n x n
    branch2c = \
        conv_bn_scale(branch2b, num_output=num_output, kernel_size=1)  # 4*base_output x n x n

    residual, residual_relu = \
        eltwize_relu(bottom, branch2c)  # 4*base_output x n x n

    return branch2a, branch2a_relu, branch2b, branch2b_relu, \
           branch2c, residual, residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64, num_output = 256):
    """

    :param stride: stride
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch1 = \
        conv_bn_scale(bottom, num_output=num_output, kernel_size=1, stride=stride)

    branch2a, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride)
    branch2b,  branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)
    branch2c = \
        conv_bn_scale(branch2b, num_output=num_output, kernel_size=1)

    residual, residual_relu = \
        eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1,  branch2a,  branch2a_relu, branch2b, \
           branch2b_relu, branch2c, residual, residual_relu


branch_string = 'n.res(stage)b(order)_branch2a, \
        n.res(stage)b(order)_branch2a_relu, n.res(stage)b(order)_branch2b,  \
        n.res(stage)b(order)_branch2b_relu, n.res(stage)b(order)_branch2c, \
         n.res(stage)b(order), n.res(stage)b(order)_relu = \
            residual_branch((bottom), base_output=(num), num_output=(output))'


branch_shortcut_string = 'n.res(stage)a_branch1, \
        n.res(stage)a_branch2a, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.res(stage)a_branch2b_relu, \
        n.res(stage)a_branch2c, n.res(stage)a, n.res(stage)a_relu = \
            residual_branch_shortcut((bottom), stride=(stride), base_output=(num),num_output=(output))'



class LayerCreator(object):
    def __init__(self, net, start_pool_idx = 0, start_residual_idx = 0, start_deconv_idx = 0, start_add_idx = 0):
        self.n = net
        
        self.pool_idx = start_pool_idx
        self.residual_idx = start_residual_idx
        self.deconv_idx = start_deconv_idx
        self.add_idx = start_add_idx
        self.conv_idx = 5
        self.bn_idx = 0
        self.concat_idx = 0
        self.relu_idx = 0
        self.loss_idx = 0
    
    def str_replace(self, exe_str, param_name, param_str):
        
        replace_str = "exe_str = exe_str.replace('((param_name))', str((param_str)))"
        replace_str = replace_str.replace('(param_str)', param_str)
        replace_str = replace_str.replace('(param_name)', param_name)
        # print replace_str
        exec replace_str
        return exe_str
    
    def Residual(self, layer, input, output):
        n = self.n
        stage_string = ''
        ret_string = ''
        bottm_str = 'layer'
        num = self.residual_idx
        if input != output:
            
            stage_string = branch_shortcut_string
            ret_string = 'out_layer = n.res(stage)a'
            ret_cmd = ret_string.replace('(stage)', str(num + 2))

        else:
            stage_string = branch_string
            ret_string = 'out_layer = n.res(stage)b(order)'
            ret_cmd = (ret_string.replace('(stage)', str(num + 2))).replace('(order)', str(0))

        cmd = (stage_string.replace('(stage)', str(num + 2)).replace('(bottom)', bottm_str).
                        replace('(num)', str(input)).replace('(order)', str(0)).
                        replace('(stride)', str(1))).replace('(output)', str(output))
        

        exec (cmd)
        exec (ret_cmd)

        self.residual_idx += 1
        return out_layer

    def Upsamping(self, input, num_output = 64, factor = 2):
        # kernel_size = (int)(2 * factor - factor % 2)
        kernel_size = factor
        stride = factor
        pad = 0
        deconv_string = "self.n.deconv(deconv_idx) = L.Deconvolution(input, \
        param=[dict(lr_mult=0, decay_mult=0)],\
        convolution_param=dict(num_output=(num_output), kernel_size=(kernel_size), \
                        stride=(stride), pad=(pad),\
                        group = (num_output),\
                         weight_filler=dict(type='nearest'),\
                         bias_term=False))"

        deconv_string = self.str_replace(deconv_string, 'deconv_idx', str(self.deconv_idx))
        deconv_string = self.str_replace(deconv_string, 'num_output', str(num_output))
        deconv_string = self.str_replace(deconv_string, 'pad', str(pad))
        deconv_string = self.str_replace(deconv_string, 'stride', str(stride))
        deconv_string = self.str_replace(deconv_string, 'kernel_size', str(kernel_size))

        exec deconv_string
        output = None
        val_string = 'output = self.n.deconv(deconv_idx)'
        val_string = self.str_replace(val_string, 'deconv_idx', str(self.deconv_idx))
        exec val_string

        self.deconv_idx += 1
        return output

    
    def Conv(self, input,num_output=64, kernel_size=1, stride=1, pad=0):
        conv_string = "self.n.conv(conv_idx) = L.Convolution(input, num_output=(num_output), \
                        kernel_size=(kernel_size), stride=(stride), pad=(pad),\
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],\
                         weight_filler=dict(type='gaussian', std=0.01),\
                         bias_filler=dict(type='constant', value=0))"
        conv_string = self.str_replace(conv_string, 'conv_idx', str(self.conv_idx))
        conv_string = self.str_replace(conv_string, 'num_output', str(num_output))
        conv_string = self.str_replace(conv_string, 'pad', str(pad))
        conv_string = self.str_replace(conv_string, 'stride', str(stride))
        conv_string = self.str_replace(conv_string, 'kernel_size', str(kernel_size))

        exec conv_string
        output = None
        val_string = 'output = self.n.conv(conv_idx)'
        val_string = self.str_replace(val_string, 'conv_idx', str(self.conv_idx))
        exec val_string

        self.conv_idx += 1
        return output
    
    def BatchNorm(self,input):
        bn_string = "self.n.bn(bn_idx) = L.BatchNorm(input, use_global_stats=False, in_place=True)"
        bn_string = self.str_replace(bn_string, 'bn_idx', str(self.bn_idx))
        scale_string = "self.n.scale(bn_idx) = L.Scale(input, scale_param=dict(bias_term=True), in_place=True)"
        scale_string = self.str_replace(scale_string, 'bn_idx', str(self.bn_idx))
        exec bn_string
        exec scale_string
        self.bn_idx += 1
        return input
    
    def ReLU(self, input):
        relu_string = "self.n.relu(relu_idx) = L.ReLU(input, in_place=True)"
        relu_string = self.str_replace(relu_string, 'relu_idx', str(self.relu_idx))

        exec relu_string
        self.relu_idx += 1
        return input
    
    def ConvBnReLU(self, input,num_output=64, kernel_size=1, stride=1, pad=0):
        conv = self.Conv(input, num_output, kernel_size, stride, pad)
        conv = self.BatchNorm(conv)
        conv = self.ReLU(conv)
        return conv
    
    def ConvReLU(self, input,num_output=64, kernel_size=1, stride=1, pad=0):
        conv = self.Conv(input, num_output, kernel_size, stride, pad)
       
        conv = self.ReLU(conv)
        return conv


    def Deconv(self, input,num_output=64, kernel_size=7, stride=1, pad=0):
        deconv_string = "self.n.deconv(deconv_idx) = L.Deconvolution(input, \
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],\
        convolution_param=dict(num_output=(num_output), kernel_size=(kernel_size), \
                        stride=(stride), pad=(pad),\
                         weight_filler=dict(type='xavier', std=0.01),\
                         bias_filler=dict(type='constant', value=0)))"
        
        deconv_string = self.str_replace(deconv_string, 'deconv_idx', str(self.deconv_idx))
        deconv_string = self.str_replace(deconv_string, 'num_output', str(num_output))
        deconv_string = self.str_replace(deconv_string, 'pad', str(pad))
        deconv_string = self.str_replace(deconv_string, 'stride', str(stride))
        deconv_string = self.str_replace(deconv_string, 'kernel_size', str(kernel_size))

        exec deconv_string
        output = None
        val_string = 'output = self.n.deconv(deconv_idx)'
        val_string = self.str_replace(val_string, 'deconv_idx', str(self.deconv_idx))
        exec val_string

        self.deconv_idx += 1
        return output

    
    def Add(self, l1, l2):
        add_string  = "self.n.add(add_idx) = L.Eltwise(l1, l2, eltwise_param=dict(operation=1))"
        add_string = self.str_replace(add_string,'add_idx', str(self.add_idx))

        exec add_string

        output = None
        val_string = 'output = self.n.add(add_idx)'
        val_string = self.str_replace(val_string, 'add_idx', str(self.add_idx))
        exec val_string

        self.add_idx += 1
        return output


    def Pool(self, input, kernel_size=2, stride=2, pool=P.Pooling.MAX):
        pool_string = 'self.n.pool(pool_idx) = L.Pooling(input, kernel_size=(kernel_size), stride=(stride), pool=(pool))'
        pool_string = self.str_replace(pool_string, 'pool_idx', str(self.pool_idx))
        pool_string = self.str_replace(pool_string, 'kernel_size', str(kernel_size))
        pool_string = self.str_replace(pool_string, 'stride', str(stride))
        pool_string = self.str_replace(pool_string, 'pool', str(pool))

        # print pool_string

        exec pool_string
        output = None
        val_string = 'output = self.n.pool(pool_idx)'
        val_string = self.str_replace(val_string, 'pool_idx', str(self.pool_idx))
        exec val_string

        self.pool_idx += 1

        return output
    
    def EuclideanLoss(self, l1, l2):
        eloss_string = 'self.n.loss(loss_idx) = L.EuclideanLoss(l1, l2,loss_weight=1)'
        eloss_string = self.str_replace(eloss_string,'loss_idx', str(self.loss_idx))
        exec eloss_string

        output = None
        val_string = 'output = self.n.loss(loss_idx)'
        val_string = self.str_replace(val_string, 'loss_idx', str(self.loss_idx))
        exec val_string
        self.loss_idx += 1
        return output
    
    def Concat(self, L1, L2):
        concat_string = "self.n.concat(concat_idx)= L.Concat(L1, L2)"
        concat_string = self.str_replace(concat_string, 'concat_idx', str(self.concat_idx))

        exec concat_string

        output = None
        val_string = 'output = self.n.concat(concat_idx)'
        val_string = self.str_replace(val_string, 'concat_idx', str(self.concat_idx))
        exec val_string

        self.concat_idx += 1
        return output
    
    def Concat_3(self, L1, L2, L3):
        concat_string = "self.n.concat(concat_idx)= L.Concat(L1, L2, L3)"
        concat_string = self.str_replace(concat_string, 'concat_idx', str(self.concat_idx))

        exec concat_string

        output = None
        val_string = 'output = self.n.concat(concat_idx)'
        val_string = self.str_replace(val_string, 'concat_idx', str(self.concat_idx))
        exec val_string

        self.concat_idx += 1
        return output