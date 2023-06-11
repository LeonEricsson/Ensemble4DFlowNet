import tensorflow as tf
import h5py

class MetaPatchHandler3D():
    class H5Generator:
        def __init__(self, file, base_models):
            self.file = file
            self.base_models = base_models

        def _normalize(self, data, venc):
            return data / venc

        def __call__(self):
            with h5py.File(self.file, 'r') as hf:
                nr_patches = hf['compartment'].shape[0]
                mask = hf['mask']
                for p_idx in range(nr_patches):
                    mask, venc, compartment = [tf.convert_to_tensor(hf[ds][p_idx]) for ds in ['mask', 'venc', 'compartment']]
                    u_hr, v_hr, w_hr = [self._normalize(tf.convert_to_tensor(hf[ds][p_idx]), venc) for ds in ['u_hr', 'v_hr', 'w_hr']]
                    uhv_m = []
                    for m_idx in range(self.base_models):
                        uhv_m.append(self._normalize(hf[f'u_m{m_idx}.0'][p_idx], venc))
                        uhv_m.append(self._normalize(hf[f'v_m{m_idx}.0'][p_idx], venc))
                        uhv_m.append(self._normalize(hf[f'w_m{m_idx}.0'][p_idx], venc))
                    yield tf.convert_to_tensor(uhv_m), u_hr, v_hr, w_hr, venc, mask, compartment

    # constructor
    def __init__(self, file, patch_size, batch_size, base_models):
        self.file = file
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.base_models = base_models
        self.velocity_fields = 3 # U, V, W
        
    def _normalize(self, data, venc):
        return data / venc

    def load_patches_from_h5(self, idx):
        with h5py.File(self.file, 'r') as hf:
            mask, venc, compartment = [tf.convert_to_tensor(hf[ds][idx]) for ds in ['mask', 'venc', 'compartment']]
            u_hr, v_hr, w_hr = [self._normalize(tf.convert_to_tensor(hf[ds][idx]), venc) for ds in ['u_hr', 'v_hr', 'w_hr']]
            uvw_m = [] 
            for m_idx in range(self.base_models): # change to number of base models
                uvw_m.append(self._normalize(hf[f'u_m{m_idx}'][idx], venc))
                uvw_m.append(self._normalize(hf[f'v_m{m_idx}'][idx], venc))
                uvw_m.append(self._normalize(hf[f'w_m{m_idx}'][idx], venc))
    
        return tf.convert_to_tensor(uvw_m), u_hr[tf.newaxis], v_hr[tf.newaxis], w_hr[tf.newaxis], venc, mask, compartment
                    
    def load_data_using_patch_index(self, idx):
        return tf.py_function(func=self.load_patches_from_h5, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[idx], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.string])
    
    def _indexes(self):
        with h5py.File(self.file, 'r') as hf:
            group = tf.convert_to_tensor(hf['mask'])
        return tf.range(group.shape[0])
        
    def initialize_dataset(self, shuffle, drop_remainder=False):
        indexes = self._indexes()
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        
        print("Total dataset:", indexes.shape[0], 'shuffle', shuffle)

        if shuffle:
            ds = ds.shuffle(buffer_size=indexes.shape[0]) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size=self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=drop_remainder)

        # prefetch, n=number of items
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def initialize_dataset2(self, shuffle, drop_remainder=False):
        '''
            Input pipeline.
        '''
        
        patch = (self.patch_size, self.patch_size, self.patch_size)
        
        ds = tf.data.Dataset.from_generator(
            self.H5Generator(self.file, self.base_models), 
            output_signature=(
                tf.TensorSpec(shape=((self.base_models*self.velocity_fields,) + patch), dtype=tf.float32, name='uhv_m'),
                tf.TensorSpec(shape=patch, dtype=tf.float32, name='u_hr'),
                tf.TensorSpec(shape=patch, dtype=tf.float32, name='v_hr'),
                tf.TensorSpec(shape=patch, dtype=tf.float32, name='w_hr'),
                tf.TensorSpec(shape=patch, dtype=tf.float32, name='mask'),
                tf.TensorSpec(shape=(), dtype=tf.float32, name='venc'),
                tf.TensorSpec(shape=(), dtype=tf.string, name='compartment')
            ))
        
        print("Total dataset:", tf.data.experimental.cardinality(ds).numpy(), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle() 
        
        ds = ds.batch(batch_size=self.batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=drop_remainder)
        
        # prefetch, n=number of items
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    

