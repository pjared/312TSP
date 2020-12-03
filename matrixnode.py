class matrixnode:
    def __init__(self, matrix, lowerBound, path):
        self.matrix = matrix
        self.lowerBound = lowerBound
        self.path = path

    def getMatrix(self):
        return self.matrix

    def getBound(self):
        return self.lowerBound

    def addPath(self, city):
        self.path.append(city)

    def getPath(self):
        return self.path