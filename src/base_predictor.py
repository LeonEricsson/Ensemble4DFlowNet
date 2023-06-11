import tensorflow as tf
import numpy as np
import time
import os
import h5py
from Network.SR4DFlowNet import SR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset


def prepare_network(patch_size, res_increase, nr_low_block, nr_hi_block, method, type_low_block, type_high_block):
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
    net = SR4DFlowNet(res_increase, method, type_high_block, type_low_block)
    prediction = net.build_network(u, v, w, u_mag, v_mag, w_mag, nr_low_block, nr_hi_block)
    model = tf.keras.Model(input_layer, prediction)

    return model


if __name__ == '__main__':
    data_dir = '../data'
    lr_filename = 'example_data_LR.h5'
    model_path = "../models/4DFlowNet-1/4DFlowNet-1-best.h5"
    
    dir = model_path.split("/")[-2]
    output_dir = f"../results"
    output_filename = 'example_data_SR.h5'
    
    # Params
    patch_size = 24
    res_increase = 2
    batch_size = 8
    round_small_values = True

    # Architectural    
    central_upsampling = 'bilinear'

    # Type and number of blocks
    type_low_block = 'resnet'
    type_hi_block = 'resnet'
    nr_low_block = 8
    nr_hi_block = 4

    # Setting up
    lr_filepath = f"{data_dir}/{lr_filename}"
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    # Check the number of rows in the file
    nr_rows = dataset.get_dataset_len(lr_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    print(f"Loading 4DFlowNet: {res_increase}x upsample")
    # Load the network
    network = prepare_network(patch_size, res_increase, nr_low_block, nr_hi_block, central_upsampling, type_low_block, type_hi_block)
    network.load_weights(model_path)

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
            sr_images = network.predict([velocities[0][patch_index],
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
                # print(f"Zero out velocity component less than {dataset.velocity_per_px}")
                # remove small velocity values
                v[np.abs(v) < dataset.velocity_per_px] = 0
            
            v = np.expand_dims(v, axis=0) 
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.velocity_colnames[i], v, compression='gzip')

        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0) 
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

    print("Done!")