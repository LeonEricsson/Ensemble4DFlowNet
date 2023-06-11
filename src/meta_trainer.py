from Network.MetaPatchHandler import MetaPatchHandler3D
from Network.MetaTrainerController import MetaTrainerController

if __name__ == "__main__":
    data_dir = '../data'
    
    # ---- Patch index files ----
    training_file = f'{data_dir}/meta_train.h5' 
    
    validate_file = f'{data_dir}/meta_validate.h5'
    
    restore = False
    if restore:
        model_dir = "../models/4DFlowNet-meta_20230411-1622"
        model_file = "4DFlowNet-meta-best.h5"
    
    # Stacking parameter
    base_models = 2

    # Hyperparameters optimisation variables
    initial_learning_rate = 1e-4
    epochs =  120
    batch_size = 32
    mask_threshold = 0.6

    # Network setting
    network_name = '4DFlowNet-meta'
    patch_size = 24

    # Residual blocks
    resblocks = 8
 
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = MetaPatchHandler3D(training_file, patch_size, batch_size, base_models)
    trainset = z.initialize_dataset(shuffle=True, drop_remainder=False)
    
    # VALIDATION iterator
    z = MetaPatchHandler3D(validate_file, patch_size, batch_size, base_models)
    valset = z.initialize_dataset(shuffle=False, drop_remainder=False)
                
    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = MetaTrainerController(patch_size, base_models, initial_learning_rate, QUICKSAVE, network_name, resblocks)
    network.init_model_dir()
    
    if restore:
        print(f"Restoring model {model_file}...")
        network.restore_model(model_dir, model_file)
        print("Learning rate", network.optimizer.lr.numpy())

    network.train_network(trainset, valset, n_epoch=epochs)
