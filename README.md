# Diabetes Prediction

## About
This is a [Streamlit](https://streamlit.io/) web application that makes predictions about **diabetes** based on the patient data provided by the user.

## How it works
- Using the **Support Vector Classifier** model provided by the [Scikit-Learn library](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), prediction about diabetes based on the patient's medical details is made.
- This model has been trained on the [Diabetes Dataset for Beginners](https://www.kaggle.com/datasets/shantanudhakadd/diabetes-dataset-for-beginners) from **Kaggle** using standardization and stratification techniques.
- The pre-trained **SVC** model has been saved as a pickle file. The app collects user inputs through a form, feeds those inputs to the model, which then makes the prediction displayed by the app as the prediction result.

### Dataset link
> [Diabetes Dataset for Beginners](diabetes.csv)

### Deployment link
> [Diabetes Prediction](https://diabetes-prediction-lr3wknr6yrdjv87jgbttka.streamlit.app/)

