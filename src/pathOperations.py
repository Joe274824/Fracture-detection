
import os
import sys

ANNOTATIONS_DIRS = ["data/nymke/RAH_annotations"]

def getAnnotationDirs():
    """
    Get the list of directories containing annotations.
    """
    paths = [f.path for f in os.scandir(ANNOTATIONS_DIRS[0]) if f.is_dir()]

    if (len(paths) > 0):
        return paths
    
    print(f"No directories found in {ANNOTATIONS_DIRS[0]}")
    sys.exit(1)

def getJsonPaths(annotationDir):
    """
    Get the list of json file paths in the given annotation directory.
    """
    jsonPaths = [f.path for f in os.scandir(annotationDir) if f.is_file() and f.name.endswith('.json')]

    if (len(jsonPaths) > 0):
        return jsonPaths
    
    print(f"No json files found in {annotationDir}")
    sys.exit(1)


def getNrrdPath(annotationDir):
    """
    Get the nrrd file path in the given annotation directory.
    """
    nrrdPaths = [f.path for f in os.scandir(annotationDir) if f.is_file() and f.name.endswith('.nrrd')]

    if (len(nrrdPaths) == 1):
        return nrrdPaths[0]
    
    print(f"Expected one nrrd file in {annotationDir}, but found {len(nrrdPaths)}")
    sys.exit(1)

if __name__ == "__main__":
    paths = getAnnotationDirs()

    print("\n3 annotation directories:", paths[:3])

    jsonPaths = getJsonPaths(paths[0])
    print("\njson paths in the first annotation directory:", jsonPaths)

    nrrdPath = getNrrdPath(paths[0])
    print("\nnrrd path in the first annotation directory:", nrrdPath)