
from jsonFileWrapper import JsonFileWrapper
from pathOperations import getAnnotationDirs, getJsonPaths, getNrrdPath
import nrrd

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
    
    def getMinMaxXYZ(self):
        spaceOrigin = self.getSpaceOrigin()
        spaceDirections = self.getSpaceDirections()
        sizes = self.getSizes()
        Xs = [spaceOrigin[0], spaceOrigin[0] + spaceDirections[0][0] * sizes[0]]
        Ys = [spaceOrigin[1], spaceOrigin[1] + spaceDirections[1][1] * sizes[1]]
        Zs = [spaceOrigin[2], spaceOrigin[2] + spaceDirections[2][2] * sizes[2]]
        return min(Xs), max(Xs), min(Ys), max(Ys), min(Zs), max(Zs)
    
    # def getRawOrigin(self):
    #     spaceOrigin = self.getSpaceOrigin()
    #     spaceDirections = self.getSpaceDirections()

    #     return [spaceOrigin[i] / spaceDirections[i][i] for i in range(3)]

def loadNrrdFile(nrrdPath):
    return nrrd.read(nrrdPath)

def loadDataForAnnotation(annotationDir):
    jsonFiles = getJsonPaths(annotationDir)
    nrrdFile = getNrrdPath(annotationDir)

    jsonWrappers = [JsonFileWrapper(file) for file in jsonFiles]
    image, header = loadNrrdFile(nrrdFile)

    return jsonWrappers, image, DataHeaderWrapper(header)

if __name__ == "__main__":
    import pprint
    annotationDir = getAnnotationDirs()[0]

    jsonWrappers, image, header = loadDataForAnnotation(annotationDir)

    print(jsonWrappers[0].getName())

    print(image.shape)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(header.getHeader())