import matplotlib.pyplot as plt
import numpy as np

def get2dSlice(image_3d, axis, slice_index):
    if axis == 0:
        return image_3d[slice_index, :, :]
    elif axis == 1:
        return image_3d[:, slice_index, :]
    elif axis == 2:
        return image_3d[:, :, slice_index]
    else:
        raise ValueError("Invalid axis in function get2dSlicce ")

def visualizeSlicesAndSave(originalImage, crop, savePath, numSlices = 20, axis=1):
    """
    Visualize 2D slices from 3D medical images along with their true and predicted segmentations
    and save the plot to a specified file path.

    Parameters:
    - originalImage: 3D numpy array representing the medical image.
    - savePath: File path to save the visualization.
    - numSlices: Number of slices to visualize. Default is 20.
    """

    gapPerSlice = int(originalImage.shape[axis] / numSlices) 
    if (gapPerSlice == 0):
        gapPerSlice = 1
    numSlices = int(originalImage.shape[axis] / gapPerSlice)

    # print("num colors", np.amin(trueSegmentation), np.amax(trueSegmentation))

    num_columns = 2
    fig, axes = plt.subplots(numSlices, num_columns, figsize=(15, 5 * numSlices))

    for i in range(numSlices):
        # Plot original image
        axes[i,0].imshow(get2dSlice(originalImage, axis, i * gapPerSlice), cmap='gray')
        axes[i,0].set_title(f"Slice {i + 1}")

    gapPerSlice = int(crop.shape[axis] / numSlices) 
    if (gapPerSlice == 0):
        gapPerSlice = 1
    numSlices = int(crop.shape[axis] / gapPerSlice)

    # print("num colors", np.amin(trueSegmentation), np.amax(trueSegmentation))

    for i in range(numSlices):
        # Plot original image
        axes[i,1].imshow(get2dSlice(crop, axis, i * gapPerSlice), cmap='gray')
        axes[i,1].set_title(f"Slice {i + 1}")

    # Adjust layout for better visualization
    plt.tight_layout()

    plt.savefig(savePath)
    
    plt.close(fig)

if __name__ == "__main__":
    print("Visualisation module")


