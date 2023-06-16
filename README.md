# Ensemble4DFlowNet :jigsaw: :mag:

Introducing Ensemble4DFlowNet: a implementation of the master's thesis, ["Generalized super-resolution of
4D Flow MRI - extending capabilities using ensemble learning"](). This project offers two frameworks for ensembled 4D Flow MRI super-resolution, utilizing [4DFlowNet](https://github.com/EdwardFerdian/4DFlowNet) as the base learner architecture. The implementation is done using Tensorflow 2.6.0 with Keras, and the necessary packages are listed in the [requirements](/requirements.txt) file.

## Example results :bar_chart:

Below are example prediction results from clinical 4D Flow MRI datasets.

Aortic arch at native resolution (3mm) super-resolved using bagging and stacking ensemble (1.5mm).
<p align="left">
    <img src="https://i.imgur.com/vCKeloZ.png" width="500">
</p>

## Training process :hammer:
Two ensemble frameworks, bagging and stacking, are available for model training and prediction. For non-ensemble 4DFlowNet instructions we refer to [4DFlowNet](https://github.com/EdwardFerdian/4DFlowNet) by Edward Ferdian.

### Prepare the data
To generate pairs of high resolution (HR) and low resolution (LR) datasets, we assume the availability of an HR CFD dataset in HDF5 format. The HR dataset should contain 3D velocity fields (u, v, w) and maximum velocity scalars (u_max, v_max, w_max). Both these quantities should be defined over time, such that u = [T, X, Y, Z] and u_max = [T, max]. Furthermore, we expect a 3D binary mask that defines the flow field regions of the data. This mask can either be static, indicated by mask = [1, X, Y, Z], or dynamic, indicated by [T, X, Y, Z]. As an example we provide /data/example_data_HR.h5

How to prepare training/validation dataset.

    1. Generate lowres dataset
        >> Configure the datapath and filenames in prepare_lowres_dataset.py
        >> Run prepare_lowres_dataset.py
        >> This will generate a separate HDF5 file for the low resolution velocity data.
    2. Generate random patches from the LR-HR dataset pairs.
        >> Configure the datapath and filenames in prepare_patches.py
        >> Configure patch_size, rotation option, and number of patches per frame
        >> Run prepare_patches.py
        >> This will generate a csv file that contains the patch information.

The patches define our model input and output, during training we forgo the notion of the complete volume and instead treat every patch as a single training sample. During prediction however (when the entire volume is of interest), we patch and stitch the volume back together before saving the predicted output. We find that this approach improves I/O performance during training.

### Isolated or Combined (non-ensemble)
Training a non-ensemble model is straightforward and is described by Edward in the original [4DFlowNet](https://github.com/EdwardFerdian/4DFlowNet) repo. However, we've made small modifications to parameters and file names so here is a quick walk through:

To train a non-ensemble 4DFlowNet model:

    1. Put all data files (HDF5) and CSV patch index files in the same directory (e.g. /data)
    2. Open base_trainer.py and configure the data_dir and the csv filenames.
    3. Adjust hyperparameters. The default values from the paper are already provided in the code.
    4. Run base_trainer.py

Adjustable parameters for base_trainer.py:

|Parameter  | Description   |
|------|--------------|
|QUICKSAVE| Option to run a "bechmark" dataset everytime a model is saved |
|benchmark_file| A patch index file (CSV) contains a list of patches. Only the first batch will be read and run into prediction. |
|restore| Option to restore model from a existing set of training weights |
| initial_learning_rate| Initial learning rate |
| epochs | number of epochs |
| batch_size| Batch size per prediction. |
| mask_threshold| Mask threshold for non-binary mask. This is used to measure relative error (accuracy) |
| network_name | The network name. The model will be saved in this name_timestamp format |
| patch_size| The image will be split into isotropic patches. Adjust according to computation power and image size. |
| res_increase| Upsample ratio. Adjustable to any integer. More upsample ratio requires more computation power. *Note*: res_increase=1 will denoise the image at the current resolution |
| central_upsampling | Which image resize method to use. Available options are bilinear, nearest, bicubic, gaussian, lanczos3 and lanczos5. Refer to tensorflow tf.image.resize documentation for more information|
| type_low_block | Which type of block architecture to use in low resolution space. Available options are resnet, dense and csp.|
| type_hi_block | Which type of block architecture to use in high resolution space. Available options are resnet, dense and csp.|
| nr_low_block | Number of blocks in low resolution space within 4DFlowNet. |
| nr_hi_block | Number of blocks in high resolution space within 4DFlowNet. |

### Stacking
For a stacking ensemble, the meta-learner needs to be trained in sequence of the base learners. The base learner accepts csv files while the meta-learner uses patched H5 files.

To train base learners and the meta-learner:

    1. Put all data files (HDF5) and CSV patch index files in the same directory (e.g. /data)
    2. Sample the patches into buckets for each base learner using one of the approaches in stacking.ipynb. We recommend approach #3. Remember that the meta-learner needs training and validation data aswell, that the base learners have not trained on!
    3. Open base_trainer.py and configure the data_dir and the csv filenames to train one base learner. Adjust available hyperparameters (detailed above)
    4. Run base_trainer.py
    5. Repeat until every base learner has been trained.
    6. Open prepare_meta_dataset.py and configure the settings.
    7. Run the file two times, first to generate the training data and then the validation data.
    8. Open meta_trainer.py, configure the settings and adjust hyperparameters.
    9. Run meta_trainer.py

Adjustable parameters for meta_trainer.py:

|Parameter  | Description   |
|------|--------------|
|restore| Option to restore model from a existing set of training weights. |
|base_models| Number of base learners. |
| initial_learning_rate| Initial learning rate |
| epochs | number of epochs |
| batch_size| Batch size per prediction.  |
| mask_threshold| Mask threshold for non-binary mask. This is used to measure relative error (accuracy) |
| network_name | The network name. The model will be saved in this name_timestamp format |
| patch_size| The image will be split into isotropic patches. Adjust according to computation power and image size. |
| resblocks | Number of residual blocks. |

### Bagging
For a bagging ensemble, we only need to train the base learners.

To train base learners:

    1. Put all data files (HDF5) and CSV patch index files in the same directory (e.g. /data)
    3. Open base_trainer.py and configure the data_dir and the csv filenames to train one base learner. Adjust available hyperparameters
    4. Run base_trainer.py

Additional adjustable parameters for bagging:

|Parameter  | Description   |
|------|--------------|
|random_sampling| Option to randomly sample instead of using all data samples |
|sample_size_fraction| How big fraction of the original data set should be considered |
|replacement| Option to perform the sampling process with replacement |

    5. Repeat until every base learner has been trained.



## Inference :crystal_ball:
To run a prediction

    1. If you have trained your own model proceed to step 4.
    2. Download our pre-trained model weights from here.
    3. Create a folder in src/models/ with the same name as your model e.g., if you downloaded the combined model you should have src/models/4DFlowNet-combined/4DFlowNet-combined-best.h5. For an ensemble you need to create a folder for each base and meta learner.
    4. Open either bagging_predictor or meta_predictor depending on the framework you use. Configure the settings.
    5. Run the file.

## Quantification 🔢

Everything used for quantitative and qualitative evaluation is present in the jupyter notebook evaluation_playground.ipynb. Certain evaluations, such as mean speed plots, relative mean error, RMSE are in a ready-to-use state but as the file suggests this is a playground file and as such certain evaluations such as the linear regression plots are less intuitive. 

### VTI Conversion
To convert h5 file to VTI:

    1. Open utils/h5_to_vti.py
    2. Configure the input directory, file name and desired output directory. 
    3. The script expects the h5 file to contain the following datasets: u, v, w, mask
    4. Run the file

