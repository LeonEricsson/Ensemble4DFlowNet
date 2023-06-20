import tensorflow as tf
import numpy as np
import time
import os
import itertools
from Network.SR4DFlowNet import SR4DFlowNet
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset
import h5py

# Weighted average layer for ensemble fusion
class WeightedAverage(tf.keras.layers.Layer):
    def __init__(self, weights) -> None:
        super().__init__()
                
    def call(self, input):
        weighted = tf.convert_to_tensor(input) * self.w
        return tf.reduce_sum(weighted, axis=[0])

def prepare_ensemble_model(base_models, meta_model):
    # All base models have the same input, extract one of them
    input = base_models[0].input
    output = [tf.unstack(model(input), axis=-1) for model in base_models]
    output = list(itertools.chain(*output)) 

    # Create the ensemble
    ensemble_output = meta_model(output)
    ensemble_model = tf.keras.Model(inputs=input, outputs=ensemble_output)
    return ensemble_model

def load_base_model(path, i):
    m = tf.keras.models.load_model(path)
    m._name = f"model_{i}"
    return m

if __name__ == '__main__':
    data_dir = '../data'
    
    LR_files = ["example"]
    
    base_model_path = "../models"
    base_model_names = ["4DFlowNet-model-1",
                        "4DFlowNet-model-2",
                        "4DFlowNet-model-3"]
    
    meta_model_path = "../models/4DFlowNet-meta/4DFlowNet-meta-best.h5"
    
    # Subsample mask
    upsample_mask = False
    
    # May need adjusting
    dir = meta_model_path.split("/")[-3]
    output_dir = f"../results/{dir}"
    
    # Params
    patch_size = 12
    res_increase = 2
    batch_size = 32
    round_small_values = True

    # Network
    low_resblock=8
    hi_resblock=4

    # Setting up
    pgen = PatchGenerator(patch_size, res_increase)
    
    print(f"Loading 4DFlowNet: {res_increase}x upsample")
    
    base_models = [load_base_model(f"{base_model_path}/{name}/{name}-best.h5", i) for i, name in enumerate(base_model_names)]
    meta_model = tf.keras.models.load_model(meta_model_path)
        
    print(f"Loaded {len(base_models)} model(s)")
    
    ensemble_model = prepare_ensemble_model(base_models, meta_model)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    for file in LR_files:
        lr_filepath = f"{data_dir}/{file}_HR.h5"
        output_filename = f'{output_dir}/{file}_SR.h5'

        dataset = ImageDataset()

        # Check the number of rows in the file
        nr_rows = dataset.get_dataset_len(lr_filepath)
        print(f"Number of rows in dataset: {nr_rows}")

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
                    # print(f"Zero out velocity component less than {dataset.velocity_per_px}")
                    # remove small velocity values
                    v[np.abs(v) < dataset.velocity_per_px] = 0
                
                v = np.expand_dims(v, axis=0) 
                prediction_utils.save_to_h5(f'{output_filename}', dataset.velocity_colnames[i], v, compression='gzip')

            if dataset.dx is not None:
                new_spacing = dataset.dx / res_increase
                new_spacing = np.expand_dims(new_spacing, axis=0) 
                prediction_utils.save_to_h5(f'{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')
        
            # Upsample mask
            if upsample_mask:
                with h5py.File(lr_filepath) as hf:
                    lr_mask = np.asarray(hf["mask"])
                    lr_mask = np.squeeze(lr_mask) # Remove single time dimension if exists
                    if lr_mask.ndim == 4:
                        lr_mask = lr_mask[nrow]
                assert(lr_mask.ndim == 3)
                sr_mask = np.repeat(np.repeat(np.repeat(lr_mask, 2, axis=0), 2, axis=1), 2, axis=2)
                sr_mask = np.expand_dims(sr_mask, axis=0)
                prediction_utils.save_to_h5(f'{output_filename}', "mask", sr_mask, compression='gzip')

    print("Done!")