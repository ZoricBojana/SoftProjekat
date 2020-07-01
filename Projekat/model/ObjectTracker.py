from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np

class Object():
    def __init__(self, centroid):
        self.centroid = centroid
        self.counter = 1
        self.appears = 1 # poslednje pojavljivanje

    def __str__(self):
        return self.centroid, ', ', self.counter

    def update(self, centroid):
        self.centroid = centroid
        self.counter += 1
        self.appears += 1

    def inc_counter(self):
        self.counter += 1


class ObjectTracker():
    def __init__(self):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_dissapeared = 20
        self.deregistered = OrderedDict()

    def register(self, cenroid):
        self.objects[self.nextObjectID] = Object(cenroid)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.deregistered[objectID] = self.objects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]


    def update(self, centroids):
        if(len(centroids) == 0):
            #ako je lista novih cetroida prazna, vrati dosadasnje objekte
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        if len(self.objects)==0:
            # ako trenutno ne pratimo ni jedan objekat, registruj sve ulazne centroide
            for i in range(0, len(centroids)):
                self.register(centroids[i])
                # otherwise, are are currently tracking objects so we need to
                # try to match the input centroids to existing object
                # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objects = list(self.objects.values()) # objects

            objectCentroids = []

            for obj in objects:
                objectCentroids.append(obj.centroid)

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), centroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                #self.objects[objectID] = centroids[col]
                self.objects[objectID].update(centroids[col])
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.max_dissapeared:
                        self.deregister(objectID)
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    # otherwise, if the number of input centroids is greater
                    # than the number of existing object centroids we need to
                    # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(centroids[col])
        # return the set of trackable objects
        return self.objects