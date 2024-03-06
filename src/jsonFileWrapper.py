import json


class JsonFileWrapper:
    def __init__(self, filePath):
        self.filePath = filePath
        self.name = filePath.split('/')[-1]

        # load data from filePath and store it in self.data
        with open(filePath, 'r') as file:
            self.data = json.load(file)

    def getName(self):
        return self.name.split('.')[0]

    def getData(self):
        return self.data
    
    def getBoxCenter(self):
        center = self.data["markups"][0]['center']
        return center
    
    def getBoxSize(self):
        return self.data["markups"][0]['size']
    
    def getBoxOrientation(self):
        return self.data["markups"][0]['orientation']

    def getMinMaxXYZ(self):
        center = self.getBoxCenter()
        size = self.getBoxSize()
        return center[0] - size[0]/2, center[0] + size[0]/2, center[1] - size[1]/2, center[1] + size[1]/2, center[2] - size[2]/2, center[2] + size[2]/2
    
if __name__ == "__main__":
    print("JsonFileWrapper module")
    jsonFile = JsonFileWrapper('data/nymke/Annotaties/Flinders_1_annotation/l_healthy.json')

    print("center of the box: ", jsonFile.getBoxCenter())
    print("size of the box: ", jsonFile.getBoxSize())
    print("orientation of the box: ", jsonFile.getBoxOrientation())