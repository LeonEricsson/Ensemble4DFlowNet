import numpy as np
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerController import TrainerController

def load_indexes(index_file, random_sampling=False, sample_size_fraction=1, replacement=True):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None


    # ----- BAGGING stuff ----
    # Get a fraction of complete dataset
    sample_size = int(len(indexes)*sample_size_fraction)
    
    if random_sampling:
        rng = np.random.default_rng()
        indexes = rng.choice(indexes, size=sample_size, replace=replacement).reshape(-1, indexes.shape[-1])

    return indexes

if __name__ == "__main__":
    data_dir = '../data'
    
        # ---- Patch index files ----
    training_file = f'{data_dir}/example_patches.csv'
    validate_file = f'{data_dir}/example_patches.csv'

    QUICKSAVE = False
    benchmark_file = f'{data_dir}/example_patches.csv'
    
    restore = False
    if restore:
        model_dir = "../models/4DFlowNet"
        model_file = "4DFlowNet-best.h5"

    # Hyperparameters optimisation variables
    initial_learning_rate = 1e-4
    epochs =  60
    batch_size = 64
    mask_threshold = 0.25

    # Network setting
    network_name = f'4DFlowNet-X'
    patch_size = 12
    res_increase = 2

    # Architectural    
    central_upsampling = 'bilinear'

    # Type and number of blocks
    type_low_block = 'resnet'
    type_hi_block = 'resnet'
    nr_low_block = 8
    nr_hi_block = 4

    # Bagging settings
    # Sample train/validation/test splits from the given data
    random_sampling = False
    sample_size_fraction = 1/2
    replacement = False

    # Load data file and indexes
    trainset = load_indexes(training_file, random_sampling, sample_size_fraction, replacement)

    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=None)
    
    # VALIDATION iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    valset = valdh.initialize_dataset(valset, shuffle=False, n_parallel=None)

    # # Bechmarking dataset, use to keep track of prediction progress per best model
    testset = None
    if QUICKSAVE and benchmark_file is not None:
        # WE use this bechmarking set so we can see the prediction progressing over time
        benchmark_set = load_indexes(benchmark_file)
        
        ph = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        # No shuffling, so we can save the first batch consistently
        testset = ph.initialize_dataset(benchmark_set, shuffle=False) 
                
    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerController(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, nr_low_block, nr_hi_block, 
                                type_low_block, type_hi_block, central_upsampling)
    network.init_model_dir()

    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())

    network.train_network(trainset, valset, n_epoch=epochs, testset=testset)
