# CS-Assignment
Assignment for the course Computer Science for Business Analytics at Erasmus University Rotterdam

This code contains a duplicate detection method using LSH and complete linkage agglomerative clustering. It aims to reduce the amount of computations needed in order to find the duplicates in the data set, which consists of TV data from four different web shops. 

The structure of the code is somewhat illogical, because I started the project in Jupyter Notebook, but when I wanted to implement the bootstrapping, I had to put everything in a regluar python script in order to be able to call the full algorithm as a function. 

The code is structured as follows:  
Line 1-19:          import packages  
Line 20-28:         import data and split data into X (TV titles), Y (TV model ID), 'shop' and 'Brand'  
Line 29-362:        full run of the algorithm, with inputs data_x, data_y, numOfRows (list) and distanceThresholdArray (list)  
FullAlg:      Line 35-60:          data cleaning process  
FullAlg:      Line 69-77:          creating model words  
FullAlg:      Line 84-88:          create binary matrix from model words  
FullAlg:      Line 97-108:         create signature matrix by calling minHash function  
FullAlg:      Line 113-117:        start LSH procedure for certain number of rows (loops over the r given in numOfRows   
FullAlg:      Line 139-157:        LSH where each band is hashed into a bucket  
FullAlg:      Line 163-205:        F1* calculations  
FullAlg:      Line 217-236:        Function for calculating Jaccard similarity between two items using binary vectors  
FullAlg:      Line 241-243:        Function for calculating Cosine similarity (not used in paper)  
FullAlg:      Line 253-273:        Function for creating dissimilarity matrix based on Jaccard    
FullAlg:      Line 275-290:        Function for creating dissimilarity matrix based on Cosine (not used in paper)  
FullAlg:      Line 292-296:        Function that makes clusters using Agglomerative clustering with complete linkage, and using certain threshold  
FullAlg:      Line 298-307:        Function that obtains duplicates from the clusters given by makeClusters function, and returns the sorted set  
FullAlg:      Line 310-316:        Functions that calculate all Jaccard/Cosine similarities for all pairs (not used)  
FullAlg:      Line 318:            creates Jaccard distance matrix by calling the function  
FullAlg:      Line 320-348:        Loop that calculates precision, recall and F1 for all threshold values, for a certain number of rows and prints a matrix containing the information  
FullAlg:      Line 350-361:        print statements to print PQ, PC, fraction of comparisons, precision, recall and F1 for this run of the algorithm (for a certain number of rows). Also prints time it took to run the algorithm.  
        
Line 367-390:        minHash function that takes a binary matrix and number of hashes that you want, and returns a signature matrix  
Line 392-396:        function that creates a bucket number for a certain band ([23],[45],[69] --> [234569])  
Line 399-406:        function meant for calculating the average of all rows except the last, eventually not used in algorithm or paper.  
Line 408-412:        setting parameters of distance thresholds, number of rows and number of bootstraps  
Line 414-425:        runs the full algorithm for 5 bootstraps, and prints information  

After running the algorithm for 5 bootstraps, I put all the data into an excel sheet (taking the best threshold value for each amount of rows) and took the average over the 5 bootstraps. I used these final averages to make the plots, which you can see the code for in Plots.py. The code in Plots.py is partially commented out, because I couldn't figure out how to make several plots in 1 run, so I made them one by one.   


