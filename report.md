## Report Summary


### Task 1: Design a k-NN classifier (k =5) for all ten genres. Evaluate the performance of the classification model
#### features : ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo", "Genre"] 
Do we have to implement the kmeans algo ourselves ? 

### Task 2: Compare the feature distribution for the four classes; pop, disco, metal and classical. Analyze how the feature distribution relates to the performance of your classifier 

feat 1 and 3 are very similar; feat 2 is also kinda related. compute covariance to put numbers on this 

### Task 3 : Same classifier but only 3 of the features above and an arbitrary additional; motivate the choice.

from the correlation matrix, we take the feature that has the highest correlation with the GenreID. We also want to minimize the
correlation between the 4 selected features. Thus, amongst the 4 initial features, we look at the pairwise correlations. Amongst the 
pair with the highest correlation, we remove the one with the lowest absolute correlation with GenreID. Then we select the feature 
with the highest correlation with GenreID, provided that its correlation with the 3 remaining features is not too high


Maybe we could test all the possibilities: 
       
    for each of the task 1 features :
        drop it
        for each of the dataset other features:
            add it
            fit knn
            predict test set
            store accuracy : currently added and removed feature
    print the pair with the highest accuracy

we can take the 4 givenFeatures, remove the removed of the above printed pair and add 
the added pair of the above printed pair
    

### Task 4: Free classifier.


