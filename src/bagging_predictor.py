import os
import time

import numpy as np
import tensorflow as tf
from Network.PatchGenerator import PatchGenerator
from Network.SR4DFlowNet import SR4DFlowNet
from utils import prediction_utils
from utils.ImageDataset import ImageDataset

# Weighted average layer for ensemble fusion
class WeightedAverage(tf.keras.layers.Layer):
    def __init__(self, weights) -> None:
        super().__init__()
        self.w = tf.constant(tf.reshape(weights, [-1, 1, 1, 1, 1, 1])) # Reshape for matrix multiply with model output
                
    def call(self, input):
        weighted = tf.convert_to_tensor(input) * self.w
        return tf.reduce_sum(weighted, axis=[0])

# Average layer for ensemble fusion
class Average(tf.keras.layers.Layer):
    def __init__(self, num_models) -> None:
        super().__init__()
        weights = tf.constant([1/num_models for _ in range(num_models)])
        self.w = tf.constant(tf.reshape(weights, [-1, 1, 1, 1, 1, 1])) # Reshape for matrix multiply with model output
                
    def call(self, input):
        weighted = tf.convert_to_tensor(input) * self.w
        return tf.reduce_sum(weighted, axis=[0])


def prepare_ensemble_model(base_models, fusion_layer):
    # All base models have the same input, extract one of them
    input = base_models[0].input
    output = [model(input) for model in base_models]
    
    # Create the ensemble
    ensemble_output = fusion_layer(output)
    ensemble_model = tf.keras.Model(inputs=input, outputs=ensemble_output)
    return ensemble_model

def prepare_base_model(patch_size, res_increase, low_resblock, hi_resblock, method, block_type):
    # Prepare input
    input_shape = (patch_size,patch_size,patch_size,1)
    u = tf.keras.layers.Input(shape=input_shape, name='u')
    v = tf.keras.layers.Input(shape=input_shape, name='v')
    w = tf.keras.layers.Input(shape=input_shape, name='w')

    u_mag = tf.keras.layers.Input(shape=input_shape, name='u_mag')
    v_mag = tf.keras.layers.Input(shape=input_shape, name='v_mag')
    w_mag = tf.keras.layers.Input(shape=input_shape, name='w_mag')

    input_layer = [u,v,w,u_mag, v_mag, w_mag]

    # network & output
    net = SR4DFlowNet(res_increase, method, block_type, block_type)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, low_resblock, hi_resblock)
    model = tf.keras.Model(input_layer, prediction)

    return model


# Prepare the network and set weights from saved model
def load_base_model(patch_size, res_increase, low_resblock, hi_resblock, model_path, method='bilinear', block_type='resnet'):
    model = prepare_base_model(patch_size, res_increase, low_resblock, hi_resblock, method, block_type)
    model.load_weights(model_path)
    return model

if __name__ == '__main__':
    data_dir = '../data'
    lr_filename = 'example_data_LR.h5'

    base_model_path = "../models"
    
    base_model_names = ["4DFlowNet-bagging-1",
                        "4DFlowNet-bagging-2", 
                        "4DFlowNet-bagging-3", 
                        "4DFlowNet-bagging-4", 
                        "4DFlowNet-bagging-5", 
                        "4DFlowNet-bagging-6", 
                        "4DFlowNet-bagging-7", 
                        "4DFlowNet-bagging-8", 
                        "4DFlowNet-bagging-9", 
                        "4DFlowNet-bagging-10"]
    
    
    output_dir = f"../results"
    output_filename = 'example_data_SR.h5'
    
     # Params
    patch_size = 24
    res_increase = 2
    batch_size = 16
    round_small_values = True
    
    # Network
    low_resblock=8
    hi_resblock=4

    # Setting up
    lr_filepath = f"{data_dir}/{lr_filename}"
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()
    
    # Load the models
    base_models = [load_base_model(patch_size, res_increase, low_resblock, hi_resblock, 
                            f"{base_model_path}/{name}/{name}-best.h5") for name in base_model_names]


    print(f"Loaded {len(base_models)} model(s)")
    
    # Choose fusion layer (Average standard)
    ensemble_model = prepare_ensemble_model(base_models, Average(len(base_model_names)))

    # Check the number of rows in the file
    nr_rows = dataset.get_dataset_len(lr_filepath)
    print(f"Number of rows in LR dataset: {nr_rows}")
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    # loop through all the rows in the input file
    for nrow in range(0, nr_rows):
        print("\n--------------------------")
        print(f"\nProcessed ({nrow+1}/{nr_rows}) - {time.ctime()}")
        # Load data file and indexes
        dataset.load_vectorfield(lr_filepath, nrow)
        print(f"Original image shape: {dataset.u.shape}")
        
        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        # Predict the patches
        results = np.zeros((0,patch_size*res_increase, patch_size*res_increase, patch_size*res_increase, 3))
        start_time = time.time()

        for current_idx in range(0, data_size, batch_size):
            time_taken = time.time() - start_time
            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
            # Prepare the batch to predict
            patch_index = np.index_exp[current_idx:current_idx+batch_size]
            sr_images = ensemble_model.predict([velocities[0][patch_index],
                                    velocities[1][patch_index],
                                    velocities[2][patch_index],
                                    magnitudes[0][patch_index],
                                    magnitudes[1][patch_index],
                                    magnitudes[2][patch_index]])

            results = np.append(results, sr_images, axis=0)
        # End of batch loop    
        time_taken = time.time() - start_time
        print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

        for i in range (0,3):
            v = pgen._patchup_with_overlap(results[:,:,:,:,i], pgen.nr_x, pgen.nr_y, pgen.nr_z)
            
            # Denormalized
            v = v * dataset.venc 
            if round_small_values:
                # remove small velocity values
                v[np.abs(v) < dataset.velocity_per_px] = 0
            
            v = np.expand_dims(v, axis=0) 
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.velocity_colnames[i], v, compression='gzip')

        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0) 
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

    print("Done!")
    
    
    
    
    
    
    