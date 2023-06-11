import tensorflow as tf
import numpy as np
import time
import os
from Network.PatchHandler3D import PatchHandler3D
import h5py
import sys

def save_to_h5(output_filepath, col_name, dataset, compression=None, chunks=True):
    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape, compression=compression, chunks=chunks) # gzip, compression_opts=4
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset

def save(predictions, hr, venc, mask, compartment, output_filepath, models, h5_params):
    """Save generated predictions (meta-learner input) along with the rest of the training data as a single HDF5 file"""
    
    # Base model predictions - U, V, W
    for i in range(0, predictions.shape[-1], 3):
        save_to_h5(output_filepath, f'u_m{i//3}', predictions[:,:,:,:,i], compression=h5_params["compression"], chunks=h5_params["chunks"])
        save_to_h5(output_filepath, f'v_m{i//3}', predictions[:,:,:,:,i+1], compression=h5_params["compression"], chunks=h5_params["chunks"])
        save_to_h5(output_filepath, f'w_m{i//3}', predictions[:,:,:,:,i+2], compression=h5_params["compression"], chunks=h5_params["chunks"])
        
    # Model names
    models = np.asarray(models, dtype=h5py.special_dtype(vlen=str))
    save_to_h5(output_filepath, 'base_models', models, compression=h5_params["compression"])

    # HR - U, V, W
    save_to_h5(output_filepath, 'u_hr', hr[0], compression=h5_params["compression"], chunks=h5_params["chunks"])
    save_to_h5(output_filepath, 'v_hr', hr[1], compression=h5_params["compression"], chunks=h5_params["chunks"])
    save_to_h5(output_filepath, 'w_hr', hr[2], compression=h5_params["compression"], chunks=h5_params["chunks"])
    
    # VENC, Mask, Compartment
    save_to_h5(output_filepath, 'mask', mask, compression=h5_params["compression"], chunks=h5_params["chunks"])
    save_to_h5(output_filepath, 'venc', venc, compression=h5_params["compression"])
    save_to_h5(output_filepath, 'compartment', compartment, compression='gzip')
    return

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

def load_model(model_path, model):
    if "_" in model:
        name = model[:model.rindex("_")]
    else:
        name = model
    model = tf.keras.models.load_model(f"{model_path}/{name}-best.h5")
    return  model

if __name__ == "__main__":
    
    # -- Settings --
    NUM_BASE_MODELS = 2
    
    data_dir = '../data'
    model_dir = "../models"
    stacking_dir = f"{data_dir}/stacking"
    
    output_filename = "meta_train.h5"

    # Meta training/validation data and the base learner model names
    fold_model_set = {
        "meta_train_patches.csv": ["4DFlowNet-stacking-1",
                                   "4DFlowNet-stacking-2"] 
    }
    
    # ----------------
    
    # H5 parameters
    h5_params = {"chunks": (10, 24, 24, 24), 
                 "rdcc_nbytes": 1024**2*4000, # 4GB Chunk cache memory
                 "rdcc_nslots": 1e7,
                 "compression":"lzf"}
    
    output_path = f'{stacking_dir}/{output_filename}'

    # Params
    patch_size = 12
    res_increase = 2
    batch_size = 128
    mask_threshold = 0.6
    round_small_values = True
    sr_patch_size = patch_size*res_increase

    # Network
    low_resblock=8
    hi_resblock=4
    
    # File check
    if os.path.exists(output_path):
        print(f'Error: output file [{output_path}] already exists')
        sys.exit(0)
        
    # Create h5 file
    f = h5py.File(output_path,'w', rdcc_nbytes=h5_params["rdcc_nbytes"], rdcc_nslots=h5_params["rdcc_nslots"])

    # Iterate over the data and generate predictions
    for fold_idx, (fold_name, model_names) in enumerate(fold_model_set.items()):
        fold_data = load_indexes(f"{stacking_dir}/{fold_name}")
        fold_models = [load_model(f"{model_dir}/{m}", m) for m in model_names]

        ph = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        metaset = ph.initialize_dataset(fold_data, shuffle=False, n_parallel=None, drop_remainder=True)

        num_batches = tf.data.experimental.cardinality(metaset).numpy()
        nr_samples = num_batches * batch_size
        
        
        print(f"Fold {fold_idx}, number of batches: {num_batches}")
        predictions = np.zeros((nr_samples, sr_patch_size, sr_patch_size, sr_patch_size, 3*NUM_BASE_MODELS), dtype='float32')
        hr = np.zeros((3, nr_samples, sr_patch_size, sr_patch_size, sr_patch_size), dtype='float32')
        venc = np.zeros((nr_samples), dtype='float32')
        mask = np.zeros((nr_samples, sr_patch_size, sr_patch_size, sr_patch_size), dtype='float32')
        compartment = np.zeros((nr_samples), dtype=h5py.special_dtype(vlen=str))
        
        start_time = time.time()
        
        # Iterate over the data and generate predictions
        for batch_idx, (data_batch) in enumerate(metaset): 
            fill_range = range(batch_idx*batch_size, (batch_idx*batch_size + batch_size))
                
            lr_input = data_batch[:6]
            normalized_hr = np.squeeze(np.asarray(data_batch[6:9]))
            venc[fill_range], mask[fill_range], compartment[fill_range] = (data.numpy() for data in data_batch[9:12])
            
            # Denormalize 
            hr[:, fill_range] = normalized_hr * venc[fill_range].reshape(1, -1, 1, 1, 1)
            
            batch_predictions = np.zeros((batch_size, sr_patch_size, sr_patch_size, sr_patch_size, 0))
            
            for model_idx, model in enumerate(fold_models):
                # Predict using 3D velocities and 3D magnitudes
                sr_images = model(lr_input, training=False)
                
                # Denormalize
                sr_images = sr_images * venc[fill_range].reshape(-1,1,1,1,1)
                
                # Concatenate along the channel axis
                batch_predictions = np.concatenate((batch_predictions,sr_images), axis=-1) # (Batch size, 24, 24, 24, 3 * NUM_BASE_MODELS)
            
            predictions[fill_range] = batch_predictions # (Samples, 24, 24, 24, 3 * NUM_BASE_MODELS)
            
            # Logging
            time_taken = time.time() - start_time
            print(f"\rProcessed {batch_idx}/{num_batches} Elapsed: {(time.time() - start_time):.2f} secs.", end='\r')
        time_taken = time.time() - start_time

        print(f"Processed fold {fold_idx}, Elapsed: {(time.time() - start_time):.2f} secs.")
        
        # Save the meta-learner training data patch-wise
        print("Saving fold...")
        save(predictions, hr, venc, mask, compartment, output_path, model_names, h5_params)
        print(f"Saved fold {fold_idx}, Elapsed: {(time.time() - start_time):.2f} secs.")

    print("Done!")