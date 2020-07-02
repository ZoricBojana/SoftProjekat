from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np

class Object():
    def __init__(self, centroid):
        self.centroid = centroid
        self.counter = 1
        self.appears = 1 # broj pojavljivanja

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
        self.objects = OrderedDict() # objekti koji su trenutno u razmatranju
        self.disappeared = OrderedDict() # recnik koji prati broj "nestajanja" objekta
        self.max_dissapeared = 20 # broj frame-ova na kojima se objekat ne nalazi, pre nego sto se deregistruje
        self.deregistered = OrderedDict() # objekti koji se vise ne prate

    def register(self, cenroid):
        # registrovanje novog objekta, tj objekta za koji se pretpostavlja da se nije ranije pojavljivao
        self.objects[self.nextObjectID] = Object(cenroid)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # prebacivanje objekta iz aktivnog u neaktivnog - pretpostavlja se da se objekat vise ne nalazi u segmentu
        self.deregistered[objectID] = self.objects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]


    def update(self, centroids):
        if(len(centroids) == 0):
            #ako je lista novih cetroida prazna, vrati dosadasnje objekte
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # ukoliko je dosegnut maksimalan broj nepojavljivanja
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        if len(self.objects)==0:
            # ako trenutno ne pratimo ni jedan objekat, registruj sve ulazne centroide
            for i in range(0, len(centroids)):
                self.register(centroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objects = list(self.objects.values()) # objects

            objectCentroids = []

            for obj in objects:
                objectCentroids.append(obj.centroid)

            # izracunaj distance
            D = dist.cdist(np.array(objectCentroids), centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                #self.objects[objectID] = centroids[col]
                self.objects[objectID].update(centroids[col])
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # ukoliko je broj "novih" centroida veci od broja objekata koji se prate proveri da li su objekti nestali
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_dissapeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(centroids[col])
        # vrati set objekata koji se prate
        return self.objects