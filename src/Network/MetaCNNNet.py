import tensorflow as tf

class MetaCNNNet():
    def __init__(self) -> None:
        pass
    
    def build_network(self, x, resblock=8, channel_nr=64):        
        x = tf.keras.layers.concatenate(x, axis=-1)
    
        x = conv3d(x, 3, channel_nr, 'SYMMETRIC', 'relu')
        x = conv3d(x, 3, channel_nr, 'SYMMETRIC', 'relu')

        x = conv3d(x, 1, channel_nr, 'SYMMETRIC', 'relu')
        x = conv3d(x, 3, channel_nr, 'SYMMETRIC', 'relu')
        
        rb = x
        for _ in range(resblock):
            rb = resnet_block(rb, "ResBlock", channel_nr, pad='SYMMETRIC')

        # 3 separate path version
        u_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        u_path = conv3d(u_path, 3, 1, 'SYMMETRIC', None)

        v_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        v_path = conv3d(v_path, 3, 1, 'SYMMETRIC', None)

        w_path = conv3d(rb, 3, channel_nr, 'SYMMETRIC', 'relu')
        w_path = conv3d(w_path, 3, 1, 'SYMMETRIC', None)
        
        b_out = tf.keras.layers.concatenate([u_path, v_path, w_path])

        return b_out

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
    

def resnet_block(x, block_name='ResBlock', channel_nr=64, scale = 1, pad='SAME'):
    tmp = conv3d(x, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    tmp = conv3d(tmp, kernel_size=3, filters=channel_nr, padding=pad, activation=None, use_bias=False, initialization=None)

    tmp = x + tmp * scale
    tmp = tf.keras.layers.LeakyReLU(alpha=0.2)(tmp)

    return tmp
