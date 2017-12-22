# Recommender_System_Adaptive_KNN

===
bpr-knn is a collaberative filter recommender.

The adaptive KNN method is implemented. https://arxiv.org/pdf/1205.2618.pdf

To use this library:

    pip install bpr_knn

It will automatically install ``numpy`` as well.


How to use
===

Process the users and items in format of indexes starts from 0. 
Then initialize the module by

    from bpr_knn import KNN
    bpr = KNN(#users, #items)

The constructor also has optional parameters ``lamI, lamJ, learningRate``.

Then train model by

	bpr.train(trainData, epochs, batchSize)

where the recommended format of trainData is list of user-item index tuple pairs.
 
    eg: [(0,1),(0,3),(1,1),(2,0),(2,3)] 

and epochs, batchSize are optional.

Finally make predictions by

    bpr.predictionsKNN(K, userIndex)

or

	bpr.predictionsAll(userIndex, itemIndex)