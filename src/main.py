import sys
import numpy as np
from getAflineTransformFromPoints import getAffineTransformFromPoints
from getCorners import getCorners
import SimpleITK as sitk
from loadDataForAnnotation import loadDataForAnnotation
from pathOperations import getAnnotationDirs
from scipy.ndimage import affine_transform

from visualisation import visualizeSlicesAndSave

annotationDirs = getAnnotationDirs()

for index, annotation in enumerate(annotationDirs):
    # if ("Flinders_1_annotation" not in annotation):
    #     continue

    json, img, imgHead = loadDataForAnnotation(annotation)

    for jsonFile in json:
        center = jsonFile.getBoxCenter()
        size = jsonFile.getBoxSize()

        imgMinMaxXYZ = imgHead.getMinMaxXYZ()
        imgMins = [imgMinMaxXYZ[i*2] for i in range(3)]

        orientation_matrix = np.reshape(jsonFile.getBoxOrientation(), (3, 3))
        corner_points = getCorners(center, orientation_matrix, size, imgMins)

        # for point in corner_points:
        #     print(point)

        # corners are in teh mm space, need to convert to voxel space
            
        # print("\n\n")
        voxelDirections = imgHead.getSpaceDirections()

        voxelPoints = np.array([np.dot(point, voxelDirections) for point in corner_points])

        for i in range(len(voxelPoints)):
            voxelPoints[i][1] *= -1

        if ((voxelPoints < -50).any()): 
            print("ERROR: negative voxel points")
            for point in voxelPoints:
                print(point)
            sys.exit()

        # for point in voxelPoints:
        #     print(point)

        # outputImage = np.zeros((100, 100, 100))

        voxelPoints = getCorners([i/2 for i in img.shape], np.identity(3), img.shape, [0, 0, 0])
        outPutCoordinates = getCorners([50, 50, 50], np.identity(3), [100, 100, 100], [0, 0, 0])

        for point in voxelPoints:
            print(point)

        for point in outPutCoordinates:
            print(point)

        def apply_affine_transform(image, affine_matrix):
            # Convert the affine matrix to a SimpleITK transform
            affine_transform = sitk.AffineTransform(3)
            affine_transform.SetTranslation(affine_matrix[:, 3])
            affine_transform.SetMatrix(affine_matrix[:3, :3].flatten())
            
            # Apply the transformation to the image
            transformed_image = sitk.Resample(
                sitk.GetImageFromArray(image), 
                affine_transform,
            )
            
            return transformed_image


        affine = getAffineTransformFromPoints(voxelPoints, outPutCoordinates)

        print(affine)

        img = np.expand_dims(np.array(img), 3)

        outputImg = sitk.GetArrayFromImage(apply_affine_transform(img, affine))

        print(img.shape)
        print(img.min(), img.max())
        print(outputImg.shape)
        print(outputImg.min(), outputImg.max())

        visualizeSlicesAndSave(img, outputImg, f"visualisation/img{index}-{jsonFile.getName()}.png")

        sys.exit()
