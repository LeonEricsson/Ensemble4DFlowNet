import numpy as np
import vtk
import SimpleITK as sitk
import h5py
import os
import scipy.ndimage as ndimage
import vtk.util.numpy_support as ns

def load_mhd(filename):
    '''
        https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
        This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    img = sitk.GetArrayFromImage(itkimage)
    return img


def scalar_to_vtk(scalar_array, spacing, filename):
    """This function write a VtkImageData vti file from a numpy array.

    :param array: input array
    :type array: :class:`numpy.ndarray`
    :param origin: the origin of the array
    :type origin: array like object of values
    :param spacing: the step in each dimension
    :type spacing: array like object of values
    :param filename: output filename (.vti)
    :type filename: str
    """
    origin = (0,0,0)
    original_shape = scalar_array.shape

    array = scalar_array.ravel(order='F')
    array = np.expand_dims(array, 1)

    vtkArray = ns.numpy_to_vtk(num_array=array, deep=True,
                            array_type=ns.get_vtk_array_type(array.dtype))
    vtkArray.SetName("Magnitude")

    print('arr component',vtkArray.GetNumberOfComponents())
    print('nr tuples',vtkArray.GetNumberOfTuples())

    imageData = vtk.vtkImageData()
    imageData.SetOrigin(origin)
    imageData.SetSpacing(spacing)
    imageData.SetDimensions(original_shape)
    imageData.GetPointData().SetScalars(vtkArray)
    
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()


# Based on: https://code.ornl.gov/rwp/javelin/commit/008c61d48384b975858f04de800a9556324d4d77
def vectors_to_vtk(vector_arrays, spacing, filename):
    """This function write a VtkImageData vti file from a numpy array.

    :param array: input array
    :type array: :class:`numpy.ndarray`
    :param origin: the origin of the array
    :type origin: array like object of values
    :param spacing: the step in each dimension
    :type spacing: array like object of values
    :param filename: output filename (.vti)
    :type filename: str
    """
    origin = (0,0,0)

    (u,v,w) = vector_arrays
    original_shape = u.shape

    u = u.ravel(order='F')
    v = v.ravel(order='F')
    w = w.ravel(order='F')
    array = np.stack((u,v,w), axis=1)
    # print(array.shape)

    # array is a shape of [n, 3]
    vtkArray = ns.numpy_to_vtk(num_array=array, deep=True,
                            array_type=ns.get_vtk_array_type(array.dtype))
    vtkArray.SetName("Velocity")

    

    imageData = vtk.vtkImageData()
    imageData.SetOrigin(origin)
    imageData.SetSpacing(spacing)
    imageData.SetDimensions(original_shape)
    imageData.GetPointData().SetScalars(vtkArray)
    

    u_arr = ns.numpy_to_vtk(num_array=u, deep=True,
                            array_type=ns.get_vtk_array_type(u.dtype))
    u_arr.SetName("U")
    imageData.GetPointData().AddArray(u_arr)

    v_arr = ns.numpy_to_vtk(num_array=v, deep=True,
                            array_type=ns.get_vtk_array_type(v.dtype))
    v_arr.SetName("V")
    imageData.GetPointData().AddArray(v_arr)

    w_arr = ns.numpy_to_vtk(num_array=w, deep=True,
                            array_type=ns.get_vtk_array_type(w.dtype))
    w_arr.SetName("W")
    imageData.GetPointData().AddArray(w_arr)
    
    

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()
    
def uvw_mask_to_vtk(vector_arrays, scalar_array, spacing, filename):
    """This function write a VtkImageData vti file from a numpy array.

    :param array: input array
    :type array: :class:`numpy.ndarray`
    :param origin: the origin of the array
    :type origin: array like object of values
    :param spacing: the step in each dimension
    :type spacing: array like object of values
    :param filename: output filename (.vti)
    :type filename: str
    """
    # Velocity
    origin = (0,0,0)

    (u,v,w) = vector_arrays
    original_shape = u.shape

    u = u.ravel(order='F')
    v = v.ravel(order='F')
    w = w.ravel(order='F')
    array = np.stack((u,v,w), axis=1)

    # array is a shape of [n, 3]
    vtkArray = ns.numpy_to_vtk(num_array=array, deep=True,
                            array_type=ns.get_vtk_array_type(array.dtype))
    vtkArray.SetName("Velocity")

    # Mask
    mask = scalar_array.ravel(order='F')
    mask = np.expand_dims(mask, 1)
    
    vtkMask = ns.numpy_to_vtk(num_array=mask, deep=True,
                            array_type=ns.get_vtk_array_type(array.dtype))
    vtkMask.SetName("Mask")
    
    imageData = vtk.vtkImageData()
    imageData.SetOrigin(origin)
    imageData.SetSpacing(spacing)
    imageData.SetDimensions(original_shape)
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.GetPointData().SetScalars(vtkMask)

    u_arr = ns.numpy_to_vtk(num_array=u, deep=True,
                            array_type=ns.get_vtk_array_type(u.dtype))
    u_arr.SetName("U")
    imageData.GetPointData().AddArray(u_arr)

    v_arr = ns.numpy_to_vtk(num_array=v, deep=True,
                            array_type=ns.get_vtk_array_type(v.dtype))
    v_arr.SetName("V")
    imageData.GetPointData().AddArray(v_arr)

    w_arr = ns.numpy_to_vtk(num_array=w, deep=True,
                            array_type=ns.get_vtk_array_type(w.dtype))
    w_arr.SetName("W")
    imageData.GetPointData().AddArray(w_arr)
    
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    writer.Write()


def get_vector_fields(input_filepath, columns, idx):
    with h5py.File(input_filepath, 'r') as hf:
        u = np.asarray(hf.get(columns[0])[idx])
        v = np.asarray(hf.get(columns[1])[idx])
        w = np.asarray(hf.get(columns[2])[idx])
    return u,v,w

def get_mask(input_filepath, idx):
    with h5py.File(input_filepath, 'r') as hf:
        mask = np.asarray(hf.get('mask'))
        mask = np.squeeze(mask) # Remove single time dimension if exists
        if mask.ndim == 4:
            mask = mask[idx]
    return mask

if __name__ == "__main__":
    input_dir = "../results/"
    output_dir = "../data/"

    columns = ['u', 'v', 'w']
    
    cases = ["example_data_SR.h5"]   
    
    for case in cases:
        print(f"Processing case {case}")
        input_filepath = f"{input_dir}/{case}.h5"
        output_path = f"{output_dir}/{case}"
        output_filename = f"{case}"

        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        print(f"Result will be saved to {output_path}")

        # Load HDF5
        with h5py.File(input_filepath, mode = 'r' ) as hdf5:
            data_nr = len(hdf5[columns[0]])

        with h5py.File(input_filepath, 'r') as hf:
            if "dx" in hf.keys():
                dx = np.asarray(hf.get("dx"))[0]
                # print(dx)
                spacing = (dx[0], dx[1], dx[2])
            else:
                spacing = (1.0, 1.0, 1.0)
        print(spacing)

        # Build a vtk file per time frame    
        for idx in range(0, data_nr, 1):
            print('Processing index', idx)
            
            u, v, w = get_vector_fields(input_filepath, columns, idx)
            mask = get_mask(input_filepath, idx)

            output_filepath = os.path.join(output_path, "{}_{}.vti".format(output_filename, idx))

            #vectors_to_vtk((u,v,w), spacing, output_filepath)
            uvw_mask_to_vtk((u,v,w), mask, spacing, output_filepath)
            
        print(f"Saved as {output_filepath}")
            