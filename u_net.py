import tensorflow as tf
import numpy as np
import scipy


class U_Net(tf.keras.layers.Layer):
    def __init__(self, out_chans):
        """
        This is the primary architecture of the model, that being a standard convolutional auto-encoder.
        See the associated paper for details around u-net.
        :param out_chans: this dictates the number of channels that the output will have
        """
        super(U_Net, self).__init__()

        self.l_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv0_1 = tf.keras.layers.Conv2D(32, 3, padding='same')
        self.conv0_2 = tf.keras.layers.Conv2D(32, 3, padding='same')

        self.max1 = tf.keras.layers.MaxPool2D(strides=2)

        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, padding='same')

        self.max2 = tf.keras.layers.MaxPool2D(strides=2)

        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, padding='same')

        self.max3 = tf.keras.layers.MaxPool2D(strides=2)

        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, padding='same')

        self.max4 = tf.keras.layers.MaxPool2D(strides=2)

        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, padding='same')
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, padding='same')

        # number of channels halved because of concatenation step
        self.up_conv1 = Up_Conv(256, 2)

        self.s_conv1_1 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.s_conv1_2 = tf.keras.layers.Conv2D(256, 3, padding='same')

        self.up_conv2 = Up_Conv(128, 2)

        self.s_conv2_1 = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.s_conv2_2 = tf.keras.layers.Conv2D(128, 3, padding='same')

        self.up_conv3 = Up_Conv(64, 2)

        self.s_conv3_1 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.s_conv3_2 = tf.keras.layers.Conv2D(64, 3, padding='same')

        self.up_conv4 = Up_Conv(32, 2)

        self.s_conv4_1 = tf.keras.layers.Conv2D(32, 3, padding='same')
        self.s_conv4_2 = tf.keras.layers.Conv2D(32, 3, padding='same')

        self.out_conv = tf.keras.layers.Conv2D(out_chans, 1, padding='same', activation='relu')

    def call(self, data):

        data0 = self.l_relu(self.conv0_2(self.l_relu(self.conv0_1(data))))

        data1 = self.max1(data0)
        data1 = self.l_relu(self.conv1_2(self.l_relu(self.conv1_1(data1))))

        data2 = self.max2(data1)
        data2 = self.l_relu(self.conv2_2(self.l_relu(self.conv2_1(data2))))

        data3 = self.max3(data2)
        data3 = self.l_relu(self.conv3_2(self.l_relu(self.conv3_1(data3))))

        data4 = self.max4(data3)
        data4 = self.l_relu(self.conv4_2(self.l_relu(self.conv4_1(data4))))

        s_data1 = tf.concat([self.up_conv1(data4), data3], axis=-1)
        s_data1 = self.l_relu(self.s_conv1_2(self.l_relu(self.s_conv1_1(s_data1))))

        s_data2 = tf.concat([self.up_conv2(s_data1), data2], axis=-1)
        s_data2 = self.l_relu(self.s_conv2_2(self.l_relu(self.s_conv2_1(s_data2))))

        s_data3 = tf.concat([self.up_conv3(s_data2), data1], axis=-1)
        s_data3 = self.l_relu(self.s_conv3_2(self.l_relu(self.s_conv3_1(s_data3))))

        s_data4 = tf.concat([self.up_conv4(s_data3), data0], axis=-1)
        s_data4 = self.l_relu(self.s_conv4_2(self.l_relu(self.s_conv4_1(s_data4))))

        return self.out_conv(s_data4)


class Up_Conv(tf.keras.layers.Layer):
    def __init__(self, out_chans, filter_sz):
        """
        This is the up-conv layer used in u-net, it up-samples the input and passes it through a convolution layer
        :param out_chans: number of channels for the output
        :param filter_sz: size of the filter used for convolution
        """
        super(Up_Conv, self).__init__()
        self.l_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.up_sample = tf.keras.layers.UpSampling2D()
        self.conv2d = tf.keras.layers.Conv2D(out_chans, filter_sz, padding='same')

    def call(self, data):
        up_sample = self.up_sample(data)
        return self.l_relu(self.conv2d(up_sample))





