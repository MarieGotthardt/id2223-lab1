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
categorical feature *type* into a numerical feature by using one-hot-encoding.
After having the data prepared this way, we investigated the predictive power of the different features. 
Therefore, we performed a examined the correlations within the features and dropped features that were
highly correlated 



### Model Selection and Fine-Tuning

### Data Sampling



## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

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
