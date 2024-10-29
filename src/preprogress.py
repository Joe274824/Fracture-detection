import sys
import os
import numpy as np
from loadDataForAnnotation import loadDataForAnnotation
from pathOperations import getAnnotationDirs
from scipy.ndimage import affine_transform
import nibabel as nib

annotationDirs = getAnnotationDirs()

def rotate_point(point, center, rotation_matrix):
    
    point = np.array(point)
    center = np.array(center)
    
    point_centered = point - center
    
    point_rotated = np.dot(rotation_matrix, point_centered.T).T
    
    point_final = point_rotated + center
    return point_final

def calculateVoxelSpacings(voxelDirections):
    """
    Calculate voxel spacings from voxel directions.
    
    Args:
    - voxelDirections (list): Voxel directions.
    
    Returns:
    - list: Voxel spacings.
    """
    voxelDirections = np.array(voxelDirections)
    voxelSpacings = [np.linalg.norm(voxelDirections[:, i]) for i in range(3)]
    column_signs = []
    for i in range(3):
        # find the component with the largest absolute value in each direction vector
        dominant_component = np.argmax(np.abs(voxelDirections[:, i]))
        # use the sign of that component as the sign of the entire direction vector
        sign = np.sign(voxelDirections[dominant_component, i])
        column_signs.append(sign)
    voxelSpacings = np.array(voxelSpacings) * column_signs
    return voxelSpacings

def voxel_to_physical(voxel_point, origin, voxelSpacings_p):
    voxel_point = np.array(voxel_point)
    origin = np.array(origin)
    voxelSpacings_p = np.array(voxelSpacings_p)
    
    physical_point = voxel_point * voxelSpacings_p + origin
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
    if img is None:
        continue

    for jsonFile in json:
        filename = os.path.basename(jsonFile.getName())
        last_name = os.path.basename(annotation)
        print(f"Processing {last_name}-{filename}: ")
        center = jsonFile.getBoxCenter()
        print(f"Center: {center}")
        size = jsonFile.getBoxSize()
        orientation_matrix = np.reshape(jsonFile.getBoxOrientation(), (3, 3))
        original_origin = np.array(imgHead['space origin'])
        voxelDirections = imgHead['space directions']
        voxelSpacings = calculateVoxelSpacings(voxelDirections)
        img_center = np.array(img.shape) / 2
        affine_inv = np.linalg.inv(orientation_matrix)

        # calculate the corner points of the box
        corners = getCorners(center, orientation_matrix, size)
        
        center_voxel = physical_to_voxel(center, original_origin, voxelSpacings)

        new_center_rotate_point = rotate_point(center_voxel, img_center, affine_inv)

        offset = np.dot(affine_inv, -img_center.T).T + img_center
        rotated_img = affine_transform(img.astype(np.float32), affine_inv, offset=offset, order=3)

        new_space_direction = np.dot(voxelDirections, affine_inv)
        new_voxelSpacings = calculateVoxelSpacings(new_space_direction)

        new_center_rotate_point = voxel_to_physical(new_center_rotate_point, original_origin, new_voxelSpacings)
        new_corners = getCorners(new_center_rotate_point, np.eye(3), size)
        new_corners_voxel = np.array([physical_to_voxel(corner, original_origin, new_voxelSpacings) for corner in new_corners])

        min_index = np.floor(np.array(new_corners_voxel.min(axis=0))).astype(int)
        max_index = np.ceil(np.array(new_corners_voxel.max(axis=0))).astype(int)

        for i in range(3):
            if min_index[i] < 0:
                min_index[i] = 0
            if min_index[i] > max_index[i]:
                min_index[i], max_index[i] = max_index[i], min_index[i]
        print(f"min_index: {min_index}")
        print(f"max_index: {max_index}")

        box_img = rotated_img[min_index[0]:max_index[0], min_index[1]:max_index[1], min_index[2]:max_index[2]]

        # create a new affine matrix for NIfTI saving
        u, _, vh = np.linalg.svd(new_space_direction, full_matrices=False)
        orthogonalized_directions = np.dot(u, vh)
        new_voxel_spacings = calculateVoxelSpacings(orthogonalized_directions)
        
        # adjust the orthogonalized directions to match the original voxel spacings
        adjusted_directions = orthogonalized_directions * np.array(voxelSpacings) / np.array(new_voxel_spacings)

        print("New voxel spacings:", calculateVoxelSpacings(adjusted_directions))
        affine = np.eye(4)
        affine[:3, :3] = adjusted_directions
        # affine[:3, 3] = original_origin

        nifti_image = nib.Nifti1Image(box_img, affine)
        print(f"box_img shape: {box_img.shape}")
        # save the NIfTI image to file
        if("fracture" in filename):
            nifti_image.to_filename(f"D://UOA/Research Project/Research Project/fracture/img{last_name}-{filename}.nii")
        else:
            nifti_image.to_filename(f"D://UOA/Research Project/Research Project/healthy/img{last_name}-{filename}.nii")
        print("=====================================")
sys.exit(0)