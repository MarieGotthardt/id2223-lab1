# Wine Quality Prediction

First Lab Assignment within the course ID2223 *Scalable Machine Learning and Deep Learning*

## Description

In this Lab, we created a severless ML system for predicting the quality of wines and for creating new wine samples
on a daily basis.

Interactive apps that utilize the ML system are hosted on Hugging Face and use Gradio for the user interface.

### Data Preparation and EDA
In a first step, we cleaned and prepared the data. First, we
filled missing data. In case of numerical features, we therefore randomly generated data points
within the range of the values of the respective features. For missing values of categorical data, 
on the other hand, we replaced missing data by a category corresponding to "Unknown". 
Afterwards, we checked the dataset for duplicated records and removed those. Then we transformed the 
categorical feature *type* into a numerical feature by using one-hot-encoding.

Furthermore, we re-categorized 
the target variable into categories from 0 to 4 (with a conceptual rating system of 1 to 5 stars), with the following binning of the original 0-10 labelling scheme:

0-3: 0 (1 stars)  
4: 1  (2 stars)  
5: 2  (3 stars)  
6-7: 3  (4 stars)  
8-10: 4  (5 stars)  


After having the data prepared this way, we investigated the predictive power of the different features. 
Therefore, we performed the correlations within the features and dropped features that were
highly (>0.7) correlated with at least one other feature.


### Model Selection and Fine-Tuning


### Data Sampling
For sampling a new synthetic wine, red or white is chosen randomly with equal probility as the wine type. A Gaussian Mixture Model (GMM) is then fit to the data for the chosen wine type, and a single sample is then taken from the GMM and put in a dataframe. The quality label is included in the GMM but is then rounded and cast to an int to fit the quality label classes.


## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.

See the file `requirements.txt`.


### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

```
pip install -r requirements.txt
```

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

Marie Gotthardt\
Samuel Härner



## Acknowledgments

Inspiration, code snippets, etc.
* [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
