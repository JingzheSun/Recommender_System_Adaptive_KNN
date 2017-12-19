import random
from math import exp
from math import log
from collections import defaultdict
import numpy as np
import time
import sys

class KNN(object):

    def __init__(self, numUsers, numItems, lamI = 6e-2, lamJ = 6e-3, learningRate = 0.1):
        self._numUsers = numUsers
        self._numItems = numItems
        self._lamI = lamI
        self._lamJ = lamJ
        self._learningRate = learningRate
        self._users = set()
        self._items = set()
        self._Iu = defaultdict(set)
        
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def train(self, trainData, epochs=30, batchSize=500):
        
        # correlation matrix
        self.C =np.random.rand(self._numItems, self._numItems)  
        for l in xrange(self._numItems):
            self.C[l][l] = 0
            for n in xrange(l, self._numItems):
                self.C[l][n] = self.C[n][l]
              
        # change batch_size to min(batch-size, len(train))
        if len(trainData) < batchSize:
            sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(trainData)))
            batchSize = len(trainData)
                  
        self._trainDict, self._users, self._items = self._dataPretreatment(trainData)
        N = len(trainData) * epochs
        users, pItems, nItems = self._sampling(N)
        itr = 0
        t2 = t0 = time.time()
        while (itr+1)*batchSize < N:
      
            self._mbgd(
                users[itr*batchSize: (itr+1)*batchSize],
                pItems[itr*batchSize: (itr+1)*batchSize],
                nItems[itr*batchSize: (itr+1)*batchSize]
            )
            
            itr += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.3f%% ) in %.1f seconds" %(str(itr*batchSize), 100.0 * float(itr*batchSize)/N, t2 - t0))
            sys.stderr.flush()
        if N > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %.2f samples per second\n" % (t2 - t0, N*1.0/(t2 - t0)))
            sys.stderr.flush()
            
            
    def _mbgd(self, users, pItems, nItems):
        
        prev = -2**10
        for _ in xrange(30):
            
            gradientC = defaultdict(float)
            obj = 0

            for ind in xrange(len(users)):
                u, i, j = users[ind], pItems[ind], nItems[ind]
                x_ui = sum([self.C[i][l] for l in self._Iu[u] if i != l])
                x_uj = sum([self.C[j][l] for l in self._Iu[u]])
                x_uij =  x_ui - x_uj
                
                for l in self._Iu[u]:
                    if l != i:
                        gradientC[(i,l)] += (1-self.sigmoid(x_uij)) + self._lamI * self.C[i][l]**2
                        gradientC[(l,i)] += (1-self.sigmoid(x_uij)) + self._lamI * self.C[l][i]**2
                    gradientC[(j,l)] += -(1-self.sigmoid(x_uij)) + self._lamJ * self.C[j][l]**2
                    gradientC[(l,j)] += -(1-self.sigmoid(x_uij)) + self._lamJ * self.C[l][j]**2
                    
                    obj -= 2*self._lamI * self.C[i][l]**2 + 2*self._lamJ * self.C[j][l]**2
                    
                obj += log(self.sigmoid(x_uij))
            
            #print 'OBJ: ', obj
            if prev > obj: 
                break
            prev = obj
            
            for a,b in gradientC:
                self.C[a][b] += self._learningRate * gradientC[(a,b)]
            
        #print _, '\n'
        
    def _sampling(self, N):
        
        sys.stderr.write("Generating %s random training samples\n" % str(N))
        userList = list(self._users)
        userIndex = np.random.randint(0, len(self._users), N)
        pItems, nItems = [], []
        cnt = 0
        for index in userIndex:
            u = userList[index]
            i = self._trainDict[u][np.random.randint(len(self._Iu[u]))]
            pItems.append(i)
            j = np.random.randint(self._numItems)
            while j in self._Iu[u]:
                j = np.random.randint(self._numItems)
            nItems.append(j)
            
            cnt += 1
            if not cnt %10000:
                sys.stderr.write("\rGenerated %s" %(str(cnt)))
                sys.stderr.flush()
        return userIndex, pItems, nItems

    def predictionsKNN(self, K, u):
        #slow
        if K >= self._Iu[u]:
            res = np.sum([self.C[:,l] for l in self._Iu[u]], 0)
        else:
            res = []
            for i in xrange(self._numItems):
                res.append(sum(sorted([self.C[i][l] for l in self._Iu[u]], reverse=True)[:K]))
        return res

    def predictionsAll(self, u):
        
        res = np.sum([self.C[:,l] for l in self._Iu[u]], 0)
        return res

    def prediction(self, u, i):
        
        scores = self.predictions(u)
        return scores[i] > sorted(scores)[self._numItem*0.8]

    def _dataPretreatment(self, data):
        dataDict = defaultdict(list)
        items = set()
        for u, i in data:
            self._Iu[u].add(i)
            dataDict[u].append(i)
            items.add(i)
        return dataDict, set(dataDict.keys()), items