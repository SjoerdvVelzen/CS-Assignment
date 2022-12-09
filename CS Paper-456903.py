#!/usr/bin/env python
# coding: utf-8

# # Computer Science Paper

# ## Packages downloaden en code importeren



import pandas as pd
import numpy as np
from numpy.linalg import norm
import math
import re
import time
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering

df = pd.read_excel(r'/Users/sjoerdvanvelzen/Documents/Econometrie/Master/Blok 2/Computer Science for Business Analytics/Assignment/TVs-excel.xlsx')


Xdata = (df['title'])
Ydata = (df['modelID'])
Shopdata = (df['shop'])
Branddata = (df['Brand'])


def completeRun(data_x, data_y, numOfRows, distanceThresholdArray):
    
    
    # ## Clean data
    
    
    startCleaning = time.time()
    for i in range(len(data_x)):
    
        data_x[i] = data_x[i].lower()
        data_x[i] = data_x[i].replace('-', '')
        data_x[i] = data_x[i].replace(',', '')
        data_x[i] = data_x[i].replace('.', '')
        data_x[i] = data_x[i].replace(':', '')
        data_x[i] = data_x[i].replace(';', '')
        data_x[i] = data_x[i].replace('/', '')
        data_x[i] = data_x[i].replace('newegg.com', '') #we don't want similarity between shops (as these are NOT duplicates)
        data_x[i] = data_x[i].replace('best buy', '')
        data_x[i] = data_x[i].replace('thenerds.net', '')
        data_x[i] = data_x[i].replace('hertz', 'hz')
        data_x[i] = data_x[i].replace('-hz', 'hz')
        data_x[i] = data_x[i].replace(' hz', 'hz')
        data_x[i] = data_x[i].replace('inches', 'inch')
        data_x[i] = data_x[i].replace('"', 'inch')
        data_x[i] = data_x[i].replace('-inch', 'inch')
        data_x[i] = data_x[i].replace(' inch', 'inch')
        data_x[i] = data_x[i].replace('refurbished:', '')
        data_x[i] = data_x[i].replace('refurbished', '')
        data_x[i] = data_x[i].replace('  ', ' ')
        data_x[i] = re.sub('\W+', ' ', data_x[i])
    endCleaning = time.time()
    
    data_x
    
    

    
    # ### Model words maken
    
    
    modelwords = []
    for i in range(len(data_x)):
        wordlist = data_x[i].split()
        for j in range(len(wordlist)):
            #if re.match("([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)", wordlist[j]) and not wordlist[j] in modelwords:
            if re.match("([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", wordlist[j]) and not wordlist[j] in modelwords:    
                modelwords.append(wordlist[j])
 
    modelwords
   
    
    # ### Binary matrix maken
    
    
    
    binaryMatrix = np.zeros((len(modelwords), len(data_x)))
    for i in range(len(modelwords)):
        for j in range(len(data_x)):
            if modelwords[i] in data_x[j]:
                binaryMatrix[i][j] = 1
  
      
    
    
    
    # ## Functions
     

    start = time.time()
    numofhashes = 600
    sigMatrix = minHash(binaryMatrix, numofhashes)
    sigMatrix
    minHashtime = time.time()
    
    
    
    
    minHashTotalTime = minHashtime - start
    
    sigMatrix.shape
    
    
    # ## LSH
    
    
    for rowsInput in numOfRows:
        startRowTimer = time.time()
        numrows = rowsInput #meer bands/minder rows = meer potential duplicates, minder bands/meer rows = meer false negatives(NIET GEWENST)
        numbands = int(numofhashes/numrows) #zorgen dat number of hashes altijd deelbaar is door number of bands!
    
    
    
        numbands
    
    
    
    
        #b bands consisting of r rows, number of hashes = bands x rows
        # use hash function to hash each band to a bucket, remember which items go in which buckets. 
        # if two in one bucket, they can be duplicates
        
        #Create dictionary for buckets and candidates
        buckets = {
        
            }
        candidates = {
        
            }
    
        #numofcomparisons = 0
        for j in range(np.shape(sigMatrix)[1]):      #for each column/item 
        
            for b in range(numbands):          #for each band of item j
                band = sigMatrix[math.floor(b*numrows):math.floor((b+1)*numrows),j]
                #band hashen naar buckets, bijhouden als er duplicate in die bucket zit
                bucket = bucketNumber(band)  #geeft bv "123456" als band bestaat uit [12, 345, 6]
                
                if bucket in buckets.keys():   #is het een duplicate/bestaat deze bucket al?
                    
                    if bucket in candidates.keys():   #is er eerder al een duplicate gevonden van deze bucket? 
                        candidates[bucket].append(j)   #kolom j toevoegen aan list van duplicates bij bucket 
                    else:
                        candidates.update({bucket: [buckets[bucket], j]})   #voeg de twee duplicate kolom nrs toe aan duplicate dict.
                else:    #deze bucket bestaat nog niet:
                    buckets.update({bucket: j})    #voeg toe aan buckets
            
            
        for buck in candidates.keys():   #remove  potential duplicates
            candidates[buck] = list(dict.fromkeys(candidates[buck]))
            
    
    
    
        # ### F1* calculation
        samplesize = len(data_x)
        maxComparisons = samplesize * (samplesize - 1) * 0.5
        #First make set of all real duplicates in sample
        allDuplicatesInSample = set()
        duplicateddf = 0
        for j in range(len(data_y)):
            for k in range(len(data_y)):
                if j<k:
                    if data_y[j] == data_y[k]:
                        allDuplicatesInSample.add(tuple(sorted((j, k))))
    
        allDuplicatesInSample = sorted(allDuplicatesInSample)
        duplicateddf = len(allDuplicatesInSample)
    
    
        #total number of potential duplicates:
        startDupTimer = time.time()
        totalnum = 0
        allpairs = set()
        duplicatepairs = set()
    
    #alles vergelijken, stop als het idd een duplicate is in de set
        for c in candidates.keys():
            for i in range(len(candidates[c])):
                for j in range(len(candidates[c])):
                    if i < j:
                        allpairs.add(tuple(sorted((candidates[c][i], candidates[c][j]))))
                        if data_y[candidates[c][i]] == data_y[candidates[c][j]]:      
                            duplicatepairs.add(tuple(sorted((candidates[c][i], candidates[c][j]))))
                        
        totalnum = len(duplicatepairs)
        numofcomparisons = len(allpairs)  
        endDupTimer = time.time()  
        allpairs = sorted(allpairs)
        
    
        pairQuality = totalnum / numofcomparisons
        
        numofduplicates = duplicateddf
        pairCompleteness = totalnum / numofduplicates
        fractionComparisonsLSH = numofcomparisons / maxComparisons
        
        harmonmean = 2 * pairQuality * pairCompleteness / (pairQuality + pairCompleteness)
        
    
    
        # ## Similarity uitrekenen voor alle potential duplicates
    
        
        #voor elke key in duplicates, get corresponding list. 
        #Calculate similarity between all those products and save them
    
   
    
        def jaccardSim(item1, item2):
        #jaccard sim is size of intersection / union of binary vector
            both1 = 0
            totalcount = 0
            vec1 = binaryMatrix[:,item1]
            vec2 = binaryMatrix[:,item2]
        
            for i in range(len(binaryMatrix[:,item1])):
                if vec1[i] == 1 and vec2[i] == 1:
                    both1 = both1 + 1
                    totalcount = totalcount + 1
                if vec1[i] - vec2[i] == 1 or vec1[i] - vec2[i] == -1:
                    totalcount = totalcount + 1
            if totalcount == 0:
                jaccardsimilarity = 0
            else:
                jaccardsimilarity = both1/totalcount
        
        
            return jaccardsimilarity
        
    
        
        
        def cosineSim(item1, item2):
            cosine = np.dot(binaryMatrix[:,item1], binaryMatrix[:,item2])/(norm(binaryMatrix[:,item1])*norm(binaryMatrix[:,item2]))
            return cosine
    
    
    
    
    
        allPairsList = list(allpairs) #convert to list in order to be able to use indices
    
    
    #Make distance matrices based on either jaccard or cosine similarities
        def create_distancematrix_jaccard(data_x, allPairsList):
        
            distanceMatrix = np.full((len(data_x), len(data_x)), 1000.00) #sets distance to practically infty for all pairs


            for pair in allPairsList:
        
                if (shopdata[pair[0]] == shopdata[pair[1]]):
                    distanceMatrix[pair[0]][pair[1]] = 1000.00
                    distanceMatrix[pair[1]][pair[0]] = 1000.00
                elif (branddata[pair[0]] != None and branddata[pair[1]] != None and branddata[pair[0]] != branddata[pair[1]]):
                    distanceMatrix[pair[0]][pair[1]] = 1000.00
                    distanceMatrix[pair[1]][pair[0]] = 1000.00
                else: 
                    distanceMatrix[pair[0]][pair[1]] = 1 - jaccardSim(pair[0], pair[1])
                    distanceMatrix[pair[1]][pair[0]] = 1 - jaccardSim(pair[0], pair[1])
            
        
            

            return distanceMatrix
    
        def create_distancematrix_cosine(data_x, allPairsList):
        
            distanceMatrix = np.full((len(data_x), len(data_x)), 1000) #sets distance to practically infty for all pairs 
            for pair in allPairsList:
                if (shopdata[pair[0]] == shopdata[pair[1]]):
                    distanceMatrix[pair[0]][pair[1]] = 1000.00
                    distanceMatrix[pair[1]][pair[0]] = 1000.00
                elif (branddata[pair[0]] != None and branddata[pair[1]] != None and branddata[pair[0]] != branddata[pair[1]]):
                    distanceMatrix[pair[0]][pair[1]] = 1000.00
                    distanceMatrix[pair[1]][pair[0]] = 1000.00
                else: 
                    distanceMatrix[pair[0]][pair[1]] = 1 - cosineSim(pair[0], pair[1])
                    distanceMatrix[pair[1]][pair[0]] = 1 - cosineSim(pair[0], pair[1])
            

            return distanceMatrix
    
        def makeClusters(distanceMatrix, distanceT):
            clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete', distance_threshold=distanceT, n_clusters=None)
            clustering.fit(distanceMatrix)
        
            return clustering

        def getDuplicates(clustering):
            duplicatesFound = set()
            for cluster in range(clustering.n_clusters_): #for all clusters check products in cluster
                products_in_cluster = np.where(clustering.labels_ == cluster)[0]
                if (len(products_in_cluster) > 1):
                    allCombs = list(combinations(products_in_cluster, 2))
                    for i in range(len(allCombs)):
                        duplicatesFound.add(tuple(sorted((allCombs[i]))))
                  
            return sorted(duplicatesFound)
    
    
        jaccardValues = np.zeros(len(allPairsList))
        for i in range(len(allPairsList)):
            jaccardValues[i] = jaccardSim(allPairsList[i][0], allPairsList[i][1])
        
        cosineValues = np.zeros(len(allPairsList))
        for i in range(len(allPairsList)):
            cosineValues[i] = cosineSim(allPairsList[i][0], allPairsList[i][1])
        
        dist_Matrix = create_distancematrix_jaccard(data_x, allPairsList)
        
        thresResults = np.zeros((len(distanceThresholdArray), 4))
        for i in range(len(distanceThresholdArray)):
        
            distanceThres = distanceThresholdArray[i]
            clusters_made = makeClusters(dist_Matrix, distanceThres)
            duplicatesAfterClust = getDuplicates(clusters_made)
            
            amountOfFP = 0
            amountOfTP = 0
            for dup in duplicatesAfterClust:
                if data_y[dup[0]] == data_y[dup[1]]:
                    amountOfTP += 1
                else:
                    amountOfFP += 1
            amountOfFN = duplicateddf - amountOfTP
    
            
            if amountOfTP + amountOfFP == 0:
                precision = 0
                recall = 0
                F1 = 0
            else:
                precision = amountOfTP / (amountOfTP + amountOfFP)
                recall = amountOfTP / (amountOfTP + amountOfFN)
                F1 = 2 * precision * recall / (precision + recall)
        
            
        
            thresResults[i][:] = [distanceThresholdArray[i], precision, recall, F1]
    
        endRowTimer = time.time()
        print("For r = " + str(numrows) + ":")
        print("Fraction of comparisons: " + str(100*fractionComparisonsLSH) + "%")
        print("Pair Quality: " + str(pairQuality))
        print("Pair Completeness: " + str(pairCompleteness))
        print("F1star: " + str(harmonmean))
        print("Threshold matrix: ")
        print("Precision, recall, F1 = ")
        print(thresResults)
        print("")
        print("Time for " + str(numrows) + " rows: " + str(endRowTimer - startRowTimer))
        print(" ")
    return thresResults, round(fractionComparisonsLSH, 6), round(pairQuality, 6), round(pairCompleteness, 6), round(harmonmean, 6)
    

#FUNCTIONS

def minHash(binMatrix, hashnumber):  
    #BinMatrix has a row for each model word and each column is an item
    count = 1
    num_columns = np.shape(binMatrix)[1]
    sigMatrix = np.zeros((hashnumber, num_columns))
    
    for i in range(hashnumber):
        #create hash function
        hashf = np.arange(len(binMatrix) - 1)
        np.random.shuffle(hashf)          #now hashf is vector with numbers from 0 to 1286 in random order
        
        for col in range(binMatrix.shape[1]): #iterates over all items
            for y in range(len(hashf)):
            
                if binMatrix[(hashf[y])][col] == 1:      #if there is a 1 in the row that the hashf determines
                    sigMatrix[i][col] = count
                    count = 1
                    break
                elif binMatrix[(hashf[y])][col] == 0:
                    count = count + 1
                    
               
    
    return sigMatrix

def bucketNumber(band):
    bucketNum = ""
    for i in range(len(band)):
        bucketNum = bucketNum + str(int(band[i]))
    return bucketNum


def calcAvg(matrix):
    shape = matrix.shape
    average = np.zeros((1, shape[1]))
    for i in range(shape[1]): #columns
        avgval = 0
        for j in range(shape[0]): #rows
            avgval += matrix[j][i]
        average[0][i] = avgval/(shape[0]-1)

numofbootstrap = 1
distanceThresholdArray = list(np.arange(0.05, 1, 0.1)) #all values from 0.4 to 0.8 with 0.05 stepsize
startTimeTotal = time.time()
numOfRows = [600, 300, 200, 150, 125, 100, 75, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
#numOfRows = [600, 300, 200, 150, 125, 100, 75, 60, 50, 40, 30, 25, 20, 10, 5]
        
for i in range(numofbootstrap):
    bootTimeStart = time.time()
    print("")
    print("Bootstrap number " + str(i + 1))
    print(" ")
    data_x, data_x_test, data_y, data_y_test, shopdata, shopdata_test, branddata, branddata_test = train_test_split(Xdata.array, Ydata.array, Shopdata.array, Branddata.array, test_size=0.37)
    
    #for each r, find best threshold inside function
    #need frac of comp, perf measures of lsh and threshold values in result
    thresMatrix, fracCompLSH, PQ, PC, F1star = completeRun(data_x, data_y, numOfRows, distanceThresholdArray)
    bootTimeEnd = time.time()
    print("Bootstrap time: " + str(bootTimeEnd - bootTimeStart))       
        
        

endTimeTotal = time.time()
print("Time for complete run: " + str(endTimeTotal - startTimeTotal))

