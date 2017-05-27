import caffe
from caffe import layers as L
from caffe import params as P


def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_bn_scale(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, conv_bn, conv_scale


def eltwize_relu(bottom1, bottom2):
    residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu


def residual_branch(bottom, base_output=64, num_output=256):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1)  # base_output x n x n
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)  # base_output x n x n
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=num_output, kernel_size=1)  # 4*base_output x n x n

    residual, residual_relu = \
        eltwize_relu(bottom, branch2c)  # 4*base_output x n x n

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
           branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


def residual_branch_shortcut(bottom, stride=2, base_output=64, num_output = 256):
    """

    :param stride: stride
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers
    """
    branch1, branch1_bn, branch1_scale = \
        conv_bn_scale(bottom, num_output=num_output, kernel_size=1, stride=stride)

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = \
        conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, stride=stride)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1)
    branch2c, branch2c_bn, branch2c_scale = \
        conv_bn_scale(branch2b, num_output=num_output, kernel_size=1)

    residual, residual_relu = \
        eltwize_relu(branch1, branch2c)  # 4*base_output x n x n

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
           branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, residual_relu


branch_string = 'n.res(stage)b(order)_branch2a, n.res(stage)b(order)_branch2a_bn, n.res(stage)b(order)_branch2a_scale, \
        n.res(stage)b(order)_branch2a_relu, n.res(stage)b(order)_branch2b, n.res(stage)b(order)_branch2b_bn, \
        n.res(stage)b(order)_branch2b_scale, n.res(stage)b(order)_branch2b_relu, n.res(stage)b(order)_branch2c, \
        n.res(stage)b(order)_branch2c_bn, n.res(stage)b(order)_branch2c_scale, n.res(stage)b(order), n.res(stage)b(order)_relu = \
            residual_branch((bottom), base_output=(num), num_output=(output))'


branch_shortcut_string = 'n.res(stage)a_branch1, n.res(stage)a_branch1_bn, n.res(stage)a_branch1_scale, \
        n.res(stage)a_branch2a, n.res(stage)a_branch2a_bn, n.res(stage)a_branch2a_scale, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.res(stage)a_branch2b_bn, n.res(stage)a_branch2b_scale, n.res(stage)a_branch2b_relu, \
        n.res(stage)a_branch2c, n.res(stage)a_branch2c_bn, n.res(stage)a_branch2c_scale, n.res(stage)a, n.res(stage)a_relu = \
            residual_branch_shortcut((bottom), stride=(stride), base_output=(num),num_output=(output))'



class LayerCreator(object):
    def __init__(self, net, start_pool_idx = 0, start_residual_idx = 0):
        self.n = net
        
        self.pool_idx = start_pool_idx
        self.residual_idx = start_residual_idx
    
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