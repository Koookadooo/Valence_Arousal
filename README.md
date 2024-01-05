# Predicting Valence and Arousal of Music Using Various Machine Learning Techniques #

**_Valence_** is the emotional positivity or negativity evoked by a piece of music while **_Arousal_** is the intensity of the emotion evoked by a piece of music.

In this project we try to predict the valence and arousal of music given the tags that have been placed on them in spotify.

In the first Jupyter Notebook, V_A_Regression, we explore different regressor models to try to predict the Valence and Arousal.
In order do this we pulled a dataset from [Soundbendor](https://github.com/Soundbendor/cs434/raw/master/lab3-deezer.zip) and, using pandas, created dataframes from the csv files withing the linked file.
This dataset contains the spotify tags and metadata from around 18000 songs. We then merged and cleaned these dataframes through feature engineering, dimentionality reduction, and imputing any NaNs.
We then began running experiements using different regression techniques along with Kfolds. In order to fine tune hyperparameters, I used CVCrossValidation.
We used Pearson R correlation scores rather than accuracy to show how well our model performs.
The benchmark score we were shooting for was a Pearson R of 0.4. I was able to exceed this score and get 0.57 using a histogram gradient boosting regressor.

In the second Jupyter Notebook, V_A_Deeplearning, we try to exceed results from the first notebook using Deep Learnin.
The same process as above was used to load, clean, and place the data into the same dataframes as the last experiment. I then attempted to engineer some more predictive features using KMeans Clustering.
After finding some realtively decent clusters given the data, I turned those clusters into features and added them back into my dataframes. I then built various deep learning models and ran the experiment.
While the results of this exceeded the Pearson R correlation benchmark of 0.4, they did not meet the results I saw from my first experiment. 

Both of the notebooks have a main workspace for the experiment followed by a report discussing the experiments and results.
