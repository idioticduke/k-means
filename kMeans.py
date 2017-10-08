import numpy as np
import support

def kMeans(dataSet, k, distMeas=support.distEclud, createCent=support.randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                #print(distJI)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print (centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

datMat = np.mat(support.loadDataSet('F:/计算机/machinelearninginaction随书源代码/Ch10/testSet.txt'))
myCentroids, clustAssing = kMeans(datMat,4)
print (myCentroids, clustAssing)