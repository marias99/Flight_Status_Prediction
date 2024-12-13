# Flight Status Prediction

This project analyzes and predicts flight cancellations using historical flight data from 2018–2022. By leveraging data processing, feature engineering, and machine learning techniques, this project provides insights into the key factors influencing flight cancellations.

---

## **Project Description**
The main objective of this project is to build a classification model to predict whether a flight will be canceled. Insights from the model can help airlines optimize their operations, reduce delays, and improve customer satisfaction. 

The dataset, sourced from Kaggle, contains over 5.6 million flight records, including details such as departure delays, arrival delays, and operational attributes.

Key highlights:
- Data preprocessing and cleaning.
- Feature engineering for improved predictive power.
- Logistic regression modeling.
- Visualizations for data insights and model results.

---

## **Repository Structure**

### Folders:
- **`notebooks/`**: Contains Jupyter Notebooks for various project steps.
  - `encoding.ipynb`: Encoding of categorical features.
  - `feature_engineering.ipynb`: Feature engineering steps.
  - `missing_values_handled.ipynb`: Handling missing values.
  - `model_training.ipynb`: Training the logistic regression model.
  - `visualizations.ipynb`: Generating visualizations for the project report.
- **`scripts/`**: Contains standalone Python scripts.
  - `exploratory_data_analysis.py`: Exploratory Data Analysis using Python.
### Key Files:
- **`Flight_Status_Prediction_Report.pdf`**: Final project report summarizing the pipeline, findings, and conclusions.
- **README.md**: Project description, repository structure, and instructions.

---

## **Pipeline Overview**

1. **Data Acquisition**:
   - Raw dataset downloaded from Kaggle and stored in Google Cloud Storage.
   - Dataset URL: [Flight Delay Dataset](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022).

2. **Data Preprocessing**:
   - Cleaned the dataset by handling missing values and removing irrelevant features.
   - Reduced dataset size for efficient modeling.

3. **Feature Engineering**:
   - Applied techniques like one-hot encoding, frequency encoding, and interaction term creation.

4. **Modeling**:
   - Trained a logistic regression model on an 80/20 train-test split.
   - Evaluated using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

5. **Visualizations**:
   - Created bar plots, scatter plots, and feature importance charts to communicate insights effectively.


