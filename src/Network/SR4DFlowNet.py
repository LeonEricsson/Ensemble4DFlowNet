import tensorflow as tf

interpolation_methods = {
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'gaussian': tf.image.ResizeMethod.GAUSSIAN,
    'lanczos3': tf.image.ResizeMethod.LANCZOS3,
    'lanczos5': tf.image.ResizeMethod.LANCZOS5}

class SR4DFlowNet():
    

    def __init__(self, res_increase, method, high_block,low_block):
        self.res_increase = res_increase
        self.high_block = high_block
        self.low_block = low_block
        self.interpolation_method = interpolation_methods[method]


    def build_network(self, u, v, w, u_mag, v_mag, w_mag, low_block=8, hi_block=4, channel_nr=64):

        network_blocks = {
            'resnet': resnet_block,
            'dense': dense_block,
            'csp': csp_block
        }

        channel_nr = 64

        speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
        mag = (u_mag ** 2 + v_mag ** 2 + w_mag ** 2) ** 0.5
        pcmr = mag * speed

        phase = tf.keras.layers.concatenate([u,v,w])
        pc    = tf.keras.layers.concatenate([pcmr, mag, speed])
        
        pc = conv3d(pc,3,channel_nr, 'SYMMETRIC', 'relu')
        pc = conv3d(pc,3,channel_nr, 'SYMMETRIC', 'relu')

        phase = conv3d(phase,3,channel_nr, 'SYMMETRIC', 'relu')
        phase = conv3d(phase,3,channel_nr, 'SYMMETRIC', 'relu')

        concat_layer = tf.keras.layers.concatenate([phase, pc])
        concat_layer = conv3d(concat_layer, 1, channel_nr, 'SYMMETRIC', 'relu')
        concat_layer = conv3d(concat_layer, 3, channel_nr, 'SYMMETRIC', 'relu')
        
        # res blocks
        rb = concat_layer

        rb = network_blocks[self.low_block](rb, low_block, channel_nr, pad='SYMMETRIC')
        
        rb = upsample3d(rb, self.res_increase)
        
        # refinement in HR
        rb = network_blocks[self.high_block](rb, hi_block, channel_nr, pad='SYMMETRIC')

        # 3 separate path version
        u_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        u_path = conv3d(u_path, 3, 1, 'SYMMETRIC', None)

        v_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        v_path = conv3d(v_path, 3, 1, 'SYMMETRIC', None)

        w_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        w_path = conv3d(w_path, 3, 1, 'SYMMETRIC', None)
        

        b_out = tf.keras.layers.concatenate([u_path, v_path, w_path])

        return b_out

def upsample3d(input_tensor, res_increase):
    """
        Resize the image by linearly interpolating the input
        using TF '``'resize_bilinear' function.

        :param input_tensor: 2D/3D image tensor, with shape:
            'batch, X, Y, Z, Channels'
        :return: interpolated volume

        Original source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/linear_resize.html
    """
    
    # We need this option for the bilinear resize to prevent shifting bug
    align = True 

    b_size, x_size, y_size, z_size, c_size = input_tensor.shape

    x_size_new, y_size_new, z_size_new = x_size * res_increase, y_size * res_increase, z_size * res_increase

    if res_increase == 1:
        # already in the target shape
        return input_tensor

    # resize y-z
    squeeze_b_x = tf.reshape(input_tensor, [-1, y_size, z_size, c_size], name='reshape_bx')
    #resize_b_x = tf.compat.v1.image.resize_bilinear(squeeze_b_x, [y_size_new, z_size_new], align_corners=align)
    resize_b_x = tf.image.resize(squeeze_b_x, [y_size_new, z_size_new])
    resume_b_x = tf.reshape(resize_b_x, [-1, x_size, y_size_new, z_size_new, c_size], name='resume_bx')

    # Reorient
    reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
    
    #   squeeze and 2d resize
    squeeze_b_z = tf.reshape(reoriented, [-1, y_size_new, x_size, c_size], name='reshape_bz')
    #resize_b_z = tf.compat.v1.image.resize_bilinear(squeeze_b_z, [y_size_new, x_size_new], align_corners=align)
    resize_b_z = tf.image.resize(squeeze_b_z, [y_size_new, x_size_new])
    resume_b_z = tf.reshape(resize_b_z, [-1, z_size_new, y_size_new, x_size_new, c_size], name='resume_bz')
    
    output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
    return output_tensor


def conv3d(x, kernel_size, filters, padding='SYMMETRIC', activation=None, initialization=None, use_bias=True):
    """
        Based on: https://github.com/gitlimlab/CycleGAN-Tensorflow/blob/master/ops.py
        For tf padding, refer to: https://www.tensorflow.org/api_docs/python/tf/pad

    """
    reg_l2 = tf.keras.regularizers.l2(5e-7)

    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = (kernel_size - 1) // 2
        x = tf.pad(x, [[0,0],[p,p],[p,p], [p,p],[0,0]], padding)
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    else:
        assert padding in ['SAME', 'VALID']
        x = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, kernel_initializer=initialization, use_bias=use_bias, kernel_regularizer=reg_l2)(x)
    return x
    

def resnet_block(x, num_layers, block_name='ResBlock', channel_nr=64, scale = 1, pad='SAME'):
    for _ in range(int(num_layers)):
        tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
        tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

        tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

        tmp = x + tmp * scale
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp

#copied from derek
def conv_block(x, block_name='ConvBlock', channel_nr=64, pad='SAME'):
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)
    tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp

#copied from derek
def dense_block(x, num_layers, block_name='DenseBlock', channel_nr=64, scale = 1, pad='SAME'):
    k = channel_nr//4
    for _ in range(int(num_layers)):
        output = conv_block(x, 'ConvBlock', k, 'SYMMETRIC')
        x = tf.concat([x, output], axis=-1)
    
    return x

#copied from derek
def csp_block(x, num_layers, block_name='CSPBlock', channel_nr=64, scale = 1, pad='SAME'):
    k = channel_nr//4
    tmp = x[:,:,:,:,:k]
    for _ in range(int(num_layers)):
        output = conv_block(tmp, 'ConvBlock', k, 'SYMMETRIC')
        tmp = tf.concat([tmp, output], axis=-1)
    tmp = tf.concat([x[:,:,:,:,k:], tmp], axis=-1)
    return tmp