
from jsonFileWrapper import JsonFileWrapper
from pathOperations import getAnnotationDirs, getJsonPaths, getNrrdPath, getNiftiPath
import nrrd
import numpy as np
import itertools
import nibabel as nib

class DataHeaderWrapper:
    def __init__(self, header):
        self.header = header

    def getHeader(self):
        return self.header

    def getSizes(self):
        return self.header['sizes']
    
    def getSpaceDirections(self):
        return self.header['space directions']
    
    def getSpaceOrigin(self):
        return self.header['space origin']
    
    def getSegmentExtent(self):
        return self.header['Segment0_Extent']
    
    def getMinMaxXYZ(self):
        spaceOrigin = self.getSpaceOrigin()
        spaceDirections = self.getSpaceDirections()
        sizes = self.getSizes()
        boundary_indices = np.array(list(itertools.product([0, sizes[0]-1], [0, sizes[1]-1], [0, sizes[2]-1])))

        # 计算边界体素的物理坐标
        min_coordinates = np.min(np.dot(boundary_indices, spaceDirections) + spaceOrigin, axis=0)
        max_coordinates = np.max(np.dot(boundary_indices, spaceDirections) + spaceOrigin, axis=0)
        return min_coordinates[0], max_coordinates[0], min_coordinates[1], max_coordinates[1], min_coordinates[2], max_coordinates[2]
    
    # def getRawOrigin(self):
    #     spaceOrigin = self.getSpaceOrigin()
    #     spaceDirections = self.getSpaceDirections()

    #     return [spaceOrigin[i] / spaceDirections[i][i] for i in range(3)]


def loadDataForAnnotation(annotationDir):
    jsonFiles = getJsonPaths(annotationDir)
    nrrdFile = getNrrdPath(annotationDir)
    niftiFile = getNiftiPath(annotationDir)
    if not niftiFile:
         return [], None, {}
    img = nib.load(niftiFile)
    jsonWrappers = []
    image = None
    header = {}
    nifti_data = img.get_fdata()
    segmented_image = img.get_fdata()
    if len(jsonFiles) > 0:
        jsonWrappers = [JsonFileWrapper(file) for file in jsonFiles]
    if nrrdFile:
        image, header = nrrd.read(nrrdFile)
        segmented_image = nifti_data * (image > 0)
    else:
        space_origin = get_space_origin(img.header)
        space_directions = get_space_directions(img.header)
        sizes = get_sizes(img.header)
        header['space directions'] = space_directions
        header['space origin'] = space_origin
        header['sizes'] = sizes
    
    return jsonWrappers, segmented_image, header


def get_space_origin(header):
    """获取空间原点"""
    return header['qoffset_x'], header['qoffset_y'], header['qoffset_z']

def get_space_directions(header):
    """获取空间方向"""
    return header['srow_x'][:3], header['srow_y'][:3], header['srow_z'][:3]

def get_sizes(header):
    """获取图像大小"""
    return header['dim'][1], header['dim'][2], header['dim'][3]


if __name__ == "__main__":
    import pprint
    annotationDir = getAnnotationDirs()[0]

    jsonWrappers, image, header = loadDataForAnnotation(annotationDir)

    print(jsonWrappers[0].getName())

    print(image.shape)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(header.getHeader())