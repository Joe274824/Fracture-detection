import sys
import os
import numpy as np
import SimpleITK as sitk
from loadDataForAnnotation import loadDataForAnnotation
from pathOperations import getAnnotationDirs
import nibabel as nib

annotationDirs = getAnnotationDirs()

def calculateVoxelSpacings(voxelDirections):
    """
    Calculate voxel spacings from voxel directions.
    
    Args:
    - voxelDirections (list): Voxel directions.
    
    Returns:
    - list: Voxel spacings.
    """
    voxelSpacings = [np.linalg.norm(voxelDirections[i]) for i in range(3)]
    column_signs = np.sign(np.sum(np.sign(voxelDirections), axis=0))
    voxelSpacings = np.array(voxelSpacings) * column_signs
    return voxelSpacings

def voxel_to_physical(voxel_point, origin, voxel_directions):
    voxel_point = np.array(voxel_point)
    origin = np.array(origin)
    voxel_directions = np.array(voxel_directions)
    
    physical_point = origin + np.dot(voxel_directions, voxel_point)
    return physical_point

def physical_to_voxel(physical_point, origin, voxelSpacings_p):
    physical_point = np.array(physical_point)
    origin = np.array(origin)
    voxel_point = (physical_point - origin) / voxelSpacings_p
    return voxel_point

def getCorners(center, orientation_matrix, size):
    # Ensure inputs are numpy arrays
    center = np.array(center)
    size = np.array(size)
    
    # Calculate half size for each dimension
    half_size = size / 2

    # List to store corner points
    corner_points = []
    
    # Iterate through all combinations of -1 and 1 to generate corner points
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                # Calculate corner position in the local box coordinate system
                local_corner = np.array([i * half_size[0], j * half_size[1], k * half_size[2]])
                # Transform local corner to the world coordinate system using the orientation matrix
                world_corner = center + np.dot(orientation_matrix, local_corner)
                # Adjust by the minimum image coordinates
                corner_points.append(world_corner)
    
    return np.array(corner_points)


for index, annotation in enumerate(annotationDirs):
    
    json, img, imgHead = loadDataForAnnotation(annotation)

    for jsonFile in json:
        print(imgHead)
        filename = os.path.basename(jsonFile.getName())
        center = jsonFile.getBoxCenter()
        center[1] *= -1
        print(f"center: {center}")
        size = jsonFile.getBoxSize()
        print(f"size: {size}")
        orientation_matrix = np.reshape(jsonFile.getBoxOrientation(), (3, 3))
        original_origin = np.array(imgHead['space origin'])
        original_origin[0] *= -1
        print(f"original_origin: {original_origin}")
        voxelDirections = imgHead['space directions']
        
        voxelSpacings_p = np.linalg.norm(voxelDirections, axis=1)
        voxelSpacings = calculateVoxelSpacings(voxelDirections)
        print(f"voxelSpacings:{voxelSpacings}")
        
        corners = getCorners(center, orientation_matrix, size)
        print("Corner Points:\n", corners)

        
        corners_voxel_space = [physical_to_voxel(corner, original_origin, voxelSpacings_p) for corner in corners]
        corners_voxel_space = np.array(corners_voxel_space)
        print("Corner Points in Voxel Space:\n", corners_voxel_space)

        min_index = np.round(np.array(corners_voxel_space.min(axis=0))).astype(int)
        max_index = np.round(np.array(corners_voxel_space.max(axis=0))).astype(int)

        for i in range(3):
            if min_index[i] > max_index[i]:
                min_index[i], max_index[i] = max_index[i], min_index[i]
        print(f"min_index: {min_index}")
        print(f"max_index: {max_index}")

        box_img = img[min_index[0]:max_index[0], min_index[1]:max_index[1], min_index[2]:max_index[2]]

        affine = np.eye(4)
        affine[:3, :3] = voxelDirections
        affine[:3, 3] = original_origin

        nifti_image = nib.Nifti1Image(box_img, affine)
        print(f"box_img shape: {box_img.shape}")

        nifti_image.to_filename(f"visualisation/img{index}-{filename}-norotate.nii")
sys.exit(0)