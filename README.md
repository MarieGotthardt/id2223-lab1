# Wine Quality Prediction

First Lab Assignment within the course ID2223 *Scalable Machine Learning and Deep Learning*

## Description

In this Lab, we created a severless ML system for predicting the quality of wines and for creating new wine samples
on a daily basis.

### Data Preparation and EDA
In a first step, we cleaned and prepared the data. First, we
filled missing data. In case of numerical features, we therefore randomly generated data points
within the range of the values of the respective features. For missing values of categorical data, 
on the other hand, we replaced missing data by a category corresponding to "Unknown". 
Afterwards, we checked the dataset for duplicated records and removed those. Then we transformed the 
categorical feature *type* into a numerical feature by using one-hot-encoding. Furthermore, we re-categorized 
the target variable into categories from 0 to 5. (maybe elaborate on that)
After having the data prepared this way, we investigated the predictive power of the different features. 
Therefore, we performed the correlations within the features and dropped features that were
highly (>0.7) correlated with at least one other feature.


### Model Selection and Fine-Tuning

### Data Sampling


## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.


### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Samuel Härner
Marie Gotthardt



## Acknowledgments

Inspiration, code snippets, etc.
* [Wine Qualtiy Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
