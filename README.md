# AdvancedMachineLearning

This is an attempt to re-implement the neural architecture of the article *Visual Domain Adaptation by Consensus-Based Transfer to Intermediate Domain* by Jongwon Choi, Youngjoon Choi, Jihoon Kim, Jinyeop Chang, Ilhwan Kwon, Youngjune Gwon, Seungjai Min (2020).

The main program will automatically call the function for training the neural network. Once the model is trained and save, the testing part can be uncommented.  It will load the model previously saved and test its performances on the testing set.

The performance obtained are far from the performances announced in the article, both in terms of run time and in term of accuracy. Each epoch last a bit less than 15 minutes, and the accuracy is 
