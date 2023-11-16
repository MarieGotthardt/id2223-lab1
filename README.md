# Wine Quality Prediction

First Lab Assignment within the course ID2223 *Scalable Machine Learning and Deep Learning*

## Description

In this Lab, we created a severless ML system for predicting the quality of wines and creating new wine samples
on a daily basis.

Interactive apps that utilize the ML system are hosted on Hugging Face and use Gradio for the user interface.

### Data Preparation and EDA
In a first step, we cleaned and prepared the data. First, we replaced missing data. In case of numerical features, we therefore randomly generated data points
within the range of the values of the respective features. For missing values of categorical data, 
on the other hand, we replaced missing data by a category corresponding to "Unknown". 
Afterwards, we checked the dataset for duplicated records and removed those. Then we transformed the 
categorical feature *type* into a numerical feature by using one-hot-encoding.

Furthermore, we re-categorized the target variable into categories from 0 to 4 (with a conceptual rating system of 1 to 
5 stars), with the following binning of the original 0-10 labelling scheme:

0-3: 0 (1 stars)  
4: 1  (2 stars)  
5: 2  (3 stars)  
6-7: 3  (4 stars)  
8-10: 4  (5 stars)  


After having the data prepared this way, we investigated the predictive power of the different features. 
Therefore, we examined the correlations between the features and dropped features that were
highly (>0.7) correlated with at least one other feature.

One problem we noticed during the EDA, was that there were only very few samples of certain target classes, i.e., the 
target classes were highly imbalanced. As we expected that this might have a negative effect on the classification
performance, we decided to perform data augmentation using the Synthetic Minority Over-sampling Technique (SMOTE).
This approach synthetically generates samples of the minor categories to achieve a balanced dataset. However, to avoid
having a uniform distribution after using SMOTE, we manually dropped some of the generated samples so that the resulting
distribution of the target variable would resemble the original distribution more closely. 


### Model Selection and Fine-Tuning
While this prediction task could have been modelled either as a regression or a classification problem, we decided to
treat it as a classification problem. 
For model selection, we performed local experiments with XGBoost and Random Forest Classifier (RFC) and performed a grid search
for hyperparameter tuning. For XGBoost, we performed the grid search over the hyperparameters *maximum depth*, *learning rate*
and *subsample* while for the RFC we performed the grid search over the hyperparameters *maximum features* and *maximum depth*.
Eventually, we decided to use the XGBoost model for the hopsworks model as it showed a slightly better performance in our local
experiments after hyperparameter tuning than the RFC. 


### Data Sampling
For sampling a new synthetic wine, red or white is chosen randomly with equal probability as the wine type. 
A Gaussian Mixture Model (GMM) is then fit to the data for the chosen wine type, and a single sample is then taken from 
the GMM and put in a dataframe. The quality label is included in the GMM but is then rounded and cast to an int to fit 
the quality label classes.


### Classification Results
To get a realistic assessment of the classification performance of our XGBoost classifier on the original wine data, we test it on
a 10% holdout set from the original data. The other 90% of the data is expanded with SMOTE and then adjusted in terms of class distributions.
This is demonstrated in the following Jupyter Notebook: [SMOTEbook](https://github.com/MarieGotthardt/id2223-lab1/blob/main/local_experiments/smotebook.ipynb)

An example run of this notebook gives the following results:

```
Accuracy before SMOTE:
XGBoost accuracy: 0.6941838649155723

Accuracy when trained on SMOTE data:
XGBoost accuracy: 0.6697936210131332

Accuracy when trained on adjusted SMOTE data:
XGBoost accuracy: 0.6848030018761726
```

As can be seen, the highest accuracy is achieved when just training on the original data. When trained on SMOTE data, where class labels are completely even in terms of class distribution, the accuracy is lower likely due to the model now over-predicting infrequent classes in the original data. After adjusting the SMOTE data to move the class distribution slightly closer to the original distribution; the accuracy goes up and is now closer to the accuracy possible when training on just original data.


### Discussion of our Approach



## Getting Started

### Dependencies and Installing

```
pip install -r requirements.txt
```

## Authors

Marie Gotthardt\
Samuel Härner



## References


* [Wine Quality Dataset](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
* [Paper on SMOTE](https://doi.org/10.48550/arXiv.1106.1813): 
Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. 
*Journal of artificial intelligence research*, 16, 321-357.
