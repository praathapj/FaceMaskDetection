# Food Review Sentiment Analysis

## **Business Problem**

In recent trend in world wide Lockdowns due to COVID19 outbreak, as Face Mask is became mandatory for everyone while roaming outside. Masks play a crucial role in protecting the health of individuals against respiratory diseases, as is one of the few precautions available for COVID-19 in the absence of immunization.

Is it possible to create a model to detect people wearing masks, not wearing them.

## Overview
An ML model which can analyzing hundreds of food reviews and return the percentage of Positive, Neutral and Negative reviews.

* **Deployed web app :**https://praathapj-foodreviewsentimentana-foodreviewsentiment-app-e1nreg.streamlit.app/

## Data Collection
Real world images obtained from kaggle: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset


## Modelling
1100 images for both class, obtained 99% accuracy with ImageNet pretrained model.

## Model Evaluation
* Evaluation metric is binary cross entropy.

## Improvement
* Obtain more data set with has images of 'mask not worn properly'.
* Also include images which has face covered without mask as 'No Mask'
