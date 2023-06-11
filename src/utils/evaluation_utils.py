import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import numpy as np
import datetime
from scipy import stats
from skimage import morphology
from scipy import ndimage

# Crop HR data (if necessary). Expects 3D shapes.
def crop(hr, pred):
    # We assume that if there is a mismatch it's because SR is smaller than HR.
    crop = np.asarray(hr.shape) - np.asarray(pred.shape)
    hr = hr[crop[0]//2:-crop[0]//2,:,:] if crop[0] else hr
    hr = hr[:, crop[1]//2:-crop[1]//2,:] if crop[1] else hr
    hr = hr[:, :, crop[2]//2:-crop[2]//2] if crop[2] else hr
    return hr
 

def get_slice_values(body, idx, axis='x'):
    if axis=='x':
        vals = body[idx, :, :]
    elif axis=='y':
        vals = body[:, idx, :]
    elif axis=='z':
        vals = body[:,:,idx]
    else:
        print("Error: x, y, z are available axes")
        return
    return vals

# Available vel_dirs are u, v, w.
def slice(file, frame, idx, vel_dir='u', axis='x'):

    with h5py.File(file, mode = 'r' ) as hf:
        body = np.asarray(hf[vel_dir][frame])
        sliced = get_slice_values(body, idx, axis)

    return sliced

def find_bad_pred(files, max_err_frame, slice_idx):
    fig, axes = plt.subplots(3, 3)
    directions = ['u', 'v', 'w']
    lr_file = files[0]
    hr_file = files[1]
    sr_file = files[2]
    for idx, subplot in enumerate(axes):
        sliced = slice(lr_file, max_err_frame, int(slice_idx/2),directions[idx], axis='z')

        subplot[0].imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
        subplot[0].set_axis_off()
        subplot[0].autoscale(False)


        # HR plot
        sliced = slice(hr_file, max_err_frame, slice_idx,directions[idx], axis='z')
        subplot[1].imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
        subplot[1].set_axis_off()
        subplot[1].autoscale(False)
        # SR plot
        sliced = slice(sr_file, max_err_frame, slice_idx,directions[idx], axis='z')
        subplot[2].imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
        subplot[2].set_axis_off()
        subplot[2].autoscale(False)
        

    fig.savefig("test_slice.png", transparent=False)
                
    
def generate_slice_comp(files, frame, lr_idx, fig_nr, vel_dir='u', axis='x'):
    plt.figure(fig_nr)
    fig, ((ax1, ax2, ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    ax1.set_title("LR")
    ax2.set_title("HR")
    ax3.set_title("Combined")
    ax4.set_title("Bag")
    ax5.set_title("Half-sub-bag")
    ax6.set_title("Cerebro")

    # ax4.set_title("Bilinear")
    # ax5.set_title("Cubic")
    # ax6.set_title("order5")


    lr_file = files[0]
    pred_files = files[2]

    hr_file = files[1]
    hr_idx = lr_idx*2

    #sr_file = files[2]

    # LR plot
    sliced = slice(lr_file, frame, lr_idx, vel_dir, axis)
    ax1.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
    # billinear_sliced = ndimage.zoom(sliced, 2, order=1)
    # ax4.imshow(billinear_sliced, interpolation='nearest', cmap='viridis', origin='lower')
    # cubic_sliced = ndimage.zoom(sliced, 2, order=3)
    # ax5.imshow(cubic_sliced, interpolation='nearest', cmap='viridis', origin='lower')
    # order5_sliced = ndimage.zoom(sliced, 2, order=5)
    # ax6.imshow(order5_sliced, interpolation='nearest', cmap='viridis', origin='lower')
    # HR plot
    sliced = slice(hr_file, frame, hr_idx, vel_dir, axis)
    ax2.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
    # SR plot
    sliced = slice(pred_files[0], frame, hr_idx, vel_dir, axis)
    ax3.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
    sliced = slice(pred_files[1], frame, hr_idx, vel_dir, axis)
    ax4.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
    sliced = slice(pred_files[2], frame, hr_idx, vel_dir, axis)
    ax5.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
    sliced = slice(pred_files[5], frame, hr_idx, vel_dir, axis)
    ax6.imshow(sliced, interpolation='nearest', cmap='viridis', origin='lower')
    plt.savefig("test.png")
    return fig_nr + 1


def find_start_name(haystack, needle):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start
    

def plot_relative_mean_error(relative_mean_error, N, pred_dirs, filename, fig_nr):
    print(f"Plotting relative mean error...")
    plt.figure(fig_nr)
    
    # Weird way of fidning correct name for model
    model_name_start = pred_dirs[0].find("/" ,pred_dirs[0].find("/")+1)+1

    #Adjust according to max num of models
    colors = ["red", "green", "blue", "orange", "purple", "black"]
    for idx, row in enumerate(relative_mean_error):
        if N == 1:
            plt.scatter(N, row, color=colors[idx], label=pred_dirs[idx][model_name_start:])
        else:
            plt.plot(np.arange(N), row, color=colors[idx], label=pred_dirs[idx][model_name_start:])

    plt.xlabel("Frame")
    plt.ylabel("Relative error (%)")
    plt.title(f"Relative speed error ({filename[:-6]})")
    plt.legend()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    plt.savefig(f"../evaluation/{filename[:-3]}_{timestamp}_RME.png")
    return fig_nr + 1

def plot_mean_speed_old(mean_speed, N, save_file, fig_nr):
    plt.figure(fig_nr)
    fig_nr += 1
    if N == 1:
        plt.scatter(N, mean_speed)
    else:
        plt.plot(np.arange(N), mean_speed)
    plt.xlabel("Frame")
    plt.ylabel("Avg. speed (cm/s)")
    plt.savefig(f"{save_file[:-3]}_speed.png")
    return fig_nr

def plot_mean_speed(mean_speed, N, save_file, fig_nr):
    print("Plotting average speed...")
    plt.figure(fig_nr)
    fig, ax = plt.subplots()
    filename_start = save_file.find("-") + 1
    colors = ["red", "green", "blue", "orange"]
    labels = ['$\mathregular{|V|}$', '$\mathregular{V_x}$', '$\mathregular{V_y}$', '$\mathregular{V_z}$']
    for i in range(4):
        ax.plot(tf.range(N), mean_speed[:, i], color=colors[i], label=labels[i])

    ax.set_xlabel("Frames")
    ax.set_ylabel("Avg. speed (cm/s)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"../evaluation/{save_file[filename_start:-3]}_speed.png")
    plt.show()
    return fig_nr + 1

def _reg_stats(subplot, hr_vals, sr_vals, color):
    reg = stats.linregress(hr_vals, sr_vals)
    x = np.array([-10, 10]) # Start, End point for the regression slope lines
    if reg.intercept < 0.0:
        reg_stats = f'$y = {reg.slope:.3f}x {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    else:
        reg_stats = f'$y = {reg.slope:.3f}x + {reg.intercept:.3f}$\n$r^2 = {reg.rvalue**2:.3f}$'
    subplot.plot(x, reg.intercept + reg.slope*x, color=color, linestyle='--', alpha=0.3)
    return reg_stats

def _plot_linear_regression(fig_nr, dimension, boundary_hr_vals, boundary_sr_vals, core_hr_vals, core_sr_vals, treat_equal=False):
    plt.figure(fig_nr)
    
    # Set limits and ticks
    xlim = ylim = 1.0
    plt.xlim(-xlim,xlim); plt.ylim(-ylim,ylim); plt.xticks([-xlim, xlim]); plt.yticks([-ylim, ylim])

    boundary_dotsize = 0.2
    boundary_color = "red"
    core_dotsize = 0.8
    core_color = "black"

    if treat_equal:
        boundary_dotsize = core_dotsize
        boundary_color = core_color
    
    plt.scatter(core_hr_vals, core_sr_vals, s=0.8, c=["black"])
    plt.scatter(boundary_hr_vals, boundary_sr_vals, s=boundary_dotsize, c=[boundary_color])
    
    boundary_reg_stats = _reg_stats(boundary_hr_vals, boundary_sr_vals)
    plt.text(-xlim/2, ylim/2, boundary_reg_stats, horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='red')
    core_reg_stats = _reg_stats(core_hr_vals, core_sr_vals)
    plt.text(xlim/2, -ylim/2, core_reg_stats, horizontalalignment='center', verticalalignment='top', fontsize=10, color="black")
    
    # Set title and labels
    plt.title(f"Correlation in V_{dimension}"); plt.xlabel("V_HR [m/s]"); plt.ylabel("V_SR [m/s]")

def _setup_lin_reg_plot(fig_nr, xlim, ylim, title):
    plt.figure(fig_nr)
    fig, subplots = plt.subplots(1,3,sharey=True, figsize=plt.figaspect(1/3))
    dimensions = ["x", "y", "z"]
    fig.suptitle(title, fontsize=16)
    for i, subplot in enumerate(subplots):
        subplot.set_xlim(-xlim,xlim); subplot.set_ylim(-ylim,ylim); subplot.set_xticks([-xlim, xlim]); subplot.set_yticks([-ylim, ylim])
        subplot.set_title(f"Correlation in $V_{{{dimensions[i]}}}$"); subplot.set_xlabel("$V_{HR}$ [m/s]"); subplot.set_ylabel("$V_{SR}$ [m/s]")
    
    return fig, subplots


def _plot_data(subplot, hr_vals, sr_vals, x, y, color, size, label):
    subplot.scatter(hr_vals, sr_vals, s=size, c=[color], label=label)
    stats = _reg_stats(subplot, hr_vals, sr_vals, color)
    subplot.legend(loc='lower right', fontsize=7)
    subplot.text(x, y, stats, horizontalalignment='center', verticalalignment='center', fontsize=7, color=color)
    


def _sample_hrsr(ground_truth_file, prediction_file, mask, peak_flow_idx, samples):
    # Use mask to find interesting samples
    sample_pot = np.where(mask == 1)
    rng = np.random.default_rng()
    # Sample <ratio> samples
    sample_idx = rng.choice(len(sample_pot[0]), replace=False, size=samples)

    # Get indexes
    x_idx = sample_pot[0][sample_idx]
    y_idx = sample_pot[1][sample_idx]
    z_idx = sample_pot[2][sample_idx]

    with h5py.File(prediction_file, mode = 'r' ) as hf:
        sr_u = np.asarray(hf['u'][peak_flow_idx])
        sr_u_vals = sr_u[x_idx, y_idx, z_idx]
        sr_v = np.asarray(hf['v'][peak_flow_idx])
        sr_v_vals = sr_v[x_idx, y_idx, z_idx]
        sr_w = np.asarray(hf['w'][peak_flow_idx])
        sr_w_vals = sr_w[x_idx, y_idx, z_idx]
        
    with h5py.File(ground_truth_file, mode = 'r' ) as hf:
        # Get velocity values in all directions
        hr_u = crop(np.asarray(hf['u'][peak_flow_idx]), sr_u)
        hr_u_vals = hr_u[x_idx, y_idx, z_idx]
        hr_v = crop(np.asarray(hf['v'][peak_flow_idx]), sr_v)
        hr_v_vals = hr_v[x_idx, y_idx, z_idx]
        hr_w = crop(np.asarray(hf['w'][peak_flow_idx]), sr_w)
        hr_w_vals = hr_w[x_idx, y_idx, z_idx]
        
        
    return [hr_u_vals, hr_v_vals, hr_w_vals], [sr_u_vals, sr_v_vals, sr_w_vals]
    

def draw_reg_line(ground_truth_file, prediction_dirs, peak_flow_idx, binary_mask, fig_nr, prediction_filename):
    """ Plot a linear regression between HR and predicted SR in peak flow frame """
    #
    # Parameters
    #

    # Hard coded
    boundary_voxels = 3000
    core_voxels = 3032

    core_fig_nr = fig_nr + 1
    boundary_fig_nr = fig_nr + 2
    xlim = 1.0
    ylim = 1.0

    colors = ["red", "green", "blue", "orange", "purple", "black"]
    x = -2*xlim/3
    y = [4*ylim/5, 3*ylim/5, 2*ylim/5, ylim/5, 0, -ylim/5]
    
    core_mask = morphology.binary_erosion(binary_mask)
    boundary_mask = binary_mask - core_mask

    model_name_start = prediction_dirs[0].find("/" ,prediction_dirs[0].find("/")+1)+1
    
    core_fig, core_subplots = _setup_lin_reg_plot(core_fig_nr, xlim, ylim, f"Core Voxels - {prediction_filename[:-6]}")
    boundary_fig, boundary_subplots = _setup_lin_reg_plot(boundary_fig_nr, xlim, ylim, f"Boundary Voxels - {prediction_filename[:-6]}")

    
    
    print(f"Plotting regression lines...")
    for i, prediction_dir in enumerate(prediction_dirs):
        prediction_file = f"{prediction_dir}/{prediction_filename}"
        boundary_hr, boundary_sr = _sample_hrsr(ground_truth_file, prediction_file, boundary_mask, peak_flow_idx, boundary_voxels)
        core_hr, core_sr = _sample_hrsr(ground_truth_file, prediction_file, core_mask, peak_flow_idx, core_voxels)
        _plot_data(core_subplots[0], core_hr[0], core_sr[0], x, y[i], colors[i], 0.5, label=prediction_dir[model_name_start:])
        _plot_data(core_subplots[1], core_hr[1], core_sr[1], x, y[i], colors[i], 0.5, label=prediction_dir[model_name_start:])
        _plot_data(core_subplots[2], core_hr[2], core_sr[2], x, y[i], colors[i], 0.5, label=prediction_dir[model_name_start:])
        
        _plot_data(boundary_subplots[0], boundary_hr[0], boundary_sr[0], x, y[i], colors[i], 0.5, label=prediction_dir[model_name_start:])
        _plot_data(boundary_subplots[1], boundary_hr[1], boundary_sr[1], x, y[i], colors[i], 0.5, label=prediction_dir[model_name_start:])
        _plot_data(boundary_subplots[2], boundary_hr[2], boundary_sr[2], x, y[i], colors[i], 0.5, label=prediction_dir[model_name_start:])
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    core_fig.savefig(f"../evaluation/{prediction_filename[:-3]}_core_{timestamp}_reg.png") 
    boundary_fig.savefig(f"../evaluation/{prediction_filename[:-3]}_boundary_{timestamp}_reg.png") 

    return fig_nr+2
