import sys
sys.path.append('../../../Build/x64/Release/pycaffe')

import inception_v3

# import HandHourglassNetWithoutBN as HandHourglassNet

import  HandHourglassNet

def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))


def demo():
    model = inception_v3.InceptionV3('imagenet_test_lmdb', 'imagenet_train_lmdb', 1000)

    train_proto = model.inception_v3_proto(64)
    test_proto = model.inception_v3_proto(64, phase='TEST')

    save_proto(train_proto, 'imagenet_train.prototxt')
    save_proto(test_proto, 'imagenet_test.prototxt')


if __name__ == '__main__':
    # demo()

    model = HandHourglassNet.HandHourglassNet('/home/maxiao/Works/HandPose/data/dataset_2/lmdb')
    train_proto = model.layers_proto()
    save_proto(train_proto, 'prototxt/pose_train_test.prototxt')

    model = HandHourglassNet.HandHourglassNet('/data1/maxiao/dataset/TopRHD/lmdb')
    test_proto = model.layers_proto(phase='TEST')


    save_proto(test_proto, 'prototxt/hand_test.prototxt')

    model = HandHourglassNet.HandHourglassNet('/home/maxiao/Works/HandPose/data/dataset_2/lmdb')
    train_vgg_proto = model.vgg_proto()
    save_proto(train_vgg_proto, 'prototxt/vgg_pose_train_test.prototxt')

    model = HandHourglassNet.HandHourglassNet('/home/maxiao/Works/HandPose/data/dataset_2/lmdb')
    test_vgg_proto = model.vgg_proto(phase='TEST')
    save_proto(test_vgg_proto, 'prototxt/vgg_depoly.prototxt')