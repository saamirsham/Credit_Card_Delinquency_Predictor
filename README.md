# Predicting Credit Card Delinquency

## Project Description

This project is dedicated to the development of a predictive model that anticipates **credit card delinquency**, a key risk management strategy in the financial sector. Utilizing personal data and historical credit information of cardholders, this model estimates the probability of payment defaults, thus facilitating the implementation of effective risk control measures.

The project leverages two distinct datasets:
- **Application Data**: This set includes vital client information.
- **Credit Details**: This set provides comprehensive data on credit card usage and behaviors.

By integrating and analyzing these datasets, the project provides a nuanced and sophisticated tool for anticipating and managing credit card delinquency.

# Problem Statement

In the banking sector, particularly for small and medium-sized banks, accurately assessing credit risk when onboarding new customers is a crucial yet challenging task. The resources and tools required for detailed credit history analysis are often lacking, resulting in broad-brush approaches that can overlook potential customers or accept high-risk ones.

Our AI-powered solution, tailored for these banks, employs a sophisticated machine learning model to predict credit delinquency based on a range of factors, including various delinquency periods. By leveraging the RandomForestClassifier, our model delivers high accuracy in predicting credit risk, helping banks make informed credit decisions.

With our solution, banks can enhance their credit risk assessments, reduce default rates, align their services with customer risk profiles, and minimize reliance on less comprehensive risk assessment methods. Our predictive model can be seamlessly integrated into banks' existing risk assessment workflows for smooth implementation.

Our project offers a modern, data-driven approach to credit risk assessment, helping small and medium-sized banks to compete effectively, minimize their exposure to credit risk, and drive sustainable growth.

## Data Overview

- **Application Data**: The dataset contains essential client information with over 400,000 observations.

- **Credit Data**: This dataset encompasses credit-related data of over one million observations. It contains key features such as MONTHS_BALANCE and STATUS.

The final sample for analysis contains over 36,000 unique applications that are common in both the datasets.

## Data Dictionary

| Column Name | Description | Data Type |
|-------------|-------------|----------|
| ID | Client identifcation number | int64 |
| is_delinquent | 1 is delinquent and 0 is paid off or no loan | bool |
| length_of_credit | Length of credit history in months | int64 |
| number_of_delinquent_months | Number of months with delinquency | int64 |
| average_delinquency_rate | Average delinquency rate | float64 |
| 3mo_delinquency | 1 is delinquent and 0 is paid off or no loan in the past 3 months | bool |
| FLAG_OWN_CAR | 1 indicates car ownership and 0 is no ownership | bool |
| FLAG_OWN_REALTY | 1 indicates property ownership and 0 is no ownership | bool |
| CNT_CHILDREN | Number of children | int64 |
| AMT_INCOME_TOTAL | Annual income | float64 |
| NAME_INCOME_TYPE | Category of income source | object |
| NAME_EDUCATION_TYPE | Education level | object |
| NAME_FAMILY_STATUS | Marital status | object |
| NAME_HOUSING_TYPE | Type of housing | object |
| FLAG_MOBIL | 1 indicates mobile phone ownership and 0 is no ownership | bool |
| FLAG_WORK_PHONE | 1 indicates work phone ownership and 0 is no ownership | bool |
| FLAG_PHONE | 1 indicates phone ownership and 0 is no ownership | bool |
| FLAG_EMAIL | 1 indicates email ownership and 0 is no ownership | bool |
| OCCUPATION_TYPE | Occupation | object |
| CNT_FAM_MEMBERS | Family size | int64 |
| AGE | Age | int64 |
| YEARS_EMPLOYED | Length of employment | int64 |

## Preliminary Data Analysis

- **Data Imbalance**: The dataset exhibited a substantial class imbalance, with a majority of the individuals recorded as not having any outstanding debt. This imbalance could potentially introduce bias in our predictive model, emphasizing the need for careful handling during model training. Approaches like oversampling the minority class, undersampling the majority class, or using synthetic data generation methods such as Synthetic Minority Over-sampling Technique (SMOTE) may be explored.

- **Missing Values in OCCUPATION_TYPE**: We identified that the `OCCUPATION_TYPE` feature had several missing entries. On closer examination, a portion of these missing values corresponded to the retired individuals, presumably because they are currently not associated with any occupation. However, a significant fraction of the missing values remained unexplained and would need to be addressed through appropriate data imputation strategies.

- **Anomaly in MONTHS_BALANCE**: An anomalous pattern was detected in the `MONTHS_BALANCE` feature. The data showed irregular jumps occurring approximately every six months. Understanding the nature of these anomalies and the potential impact they could have on our analysis is vital. Possible explanations might be seasonality effects or data collection issues, both of which warrant further investigation.

- **Data Consistency**: Beyond these specific points, the EDA process also helped us ensure the overall consistency of our dataset. We checked for other missing values, outliers, and potential errors, ensuring the dataset's readiness for the subsequent stages of our project.

By addressing these challenges upfront, we can improve the reliability of our findings and the predictive power of our final model.


## Data Cleaning

The data cleaning process aimed at enhancing data quality by:

- Removing duplicated IDs: The dataset contained 47 duplicated IDs. It is crucial to note that no complete duplicates were identified in the samples, rather the ID's themselves were duplicated in error. A deeper analysis revealed substantial disparities among these, deeming it unreasonable to retain such instances. 

- Handling missing values: The missing values in the OCCUPATION_TYPE feature were addressed by creating a 'missing' label.

- Dropping irrelevant features: The GENDER feature was dropped to prevent systemic bias in the model predictions. Similarly, DAYS_EMPLOYED and DAYS_BIRTH features were dropped as they did not provide valuable human-readable information.

- Generating new features: In addition to the original features, new ones were created such as is_delinquent, length_of_credit, 6mo_delinquency, 12mo_delinquency, and more.

- Merging datasets: The application and credit datasets were merged along the 36,457 unique IDs shared between both datasets to create a comprehensive dataset for further analysis.

- A custom function, credit_approval_data_cleaner, was utilized to clean the training data. The test set data was cleaned in the model evaluation stage.

## Exploratory Data Analysis with Cleaned Data

The data cleaning stage is integral to our data analysis pipeline, aiming to enhance the overall data quality and thereby improve the reliability of our model predictions. This phase encompassed several steps, each addressing specific data quality issues:

- **Removal of Duplicated IDs**: In our initial dataset, we observed 47 instances of duplicated IDs. It is important to emphasize that these were not complete row duplications, but only the IDs were duplicated. A comprehensive analysis of these entries revealed significant differences between the records sharing the same ID, indicating these were likely data entry errors rather than genuinely duplicated records. To maintain data integrity and avoid potentially skewed analyses, we chose to remove these instances from the dataset.

- **Handling Missing Values**: The `OCCUPATION_TYPE` feature was identified to have a substantial number of missing values. To address this, instead of discarding these records or filling them with a calculated value, we chose to label these missing values explicitly as 'missing'. This approach preserves the original data structure while clearly indicating the absence of information.

- **Dropping Irrelevant Features**: As part of our effort to ensure our model is ethical and fair, we decided to drop the `GENDER` feature. This decision was made to prevent our model from potentially propagating systemic bias. We also dropped the `DAYS_EMPLOYED` and `DAYS_BIRTH` features as these did not provide valuable insights in their current format and were not readily interpretable.

- **Feature Engineering**: Beyond cleaning and refining the existing data, we also generated new features to enrich our dataset and potentially enhance our model's predictive power. These new features include `is_delinquent`, `length_of_credit`, `6mo_delinquency`, `12mo_delinquency`, among others. Each of these features was designed with a specific purpose in mind, such as capturing trends over time or providing a more nuanced view of the individual's credit behavior.

- **Merging Datasets**: To create a comprehensive view of each individual's credit situation, we merged the application and credit datasets. This merge operation was performed on the 36,457 unique IDs shared between the two datasets, resulting in a unified dataset ready for further analysis.

- **Custom Function for Data Cleaning**: We developed a custom function, `credit_approval_data_cleaner`, to automate the data cleaning process. This function was applied to the training data during the data preparation stage and was later used to clean the test set data during the model evaluation stage.

Through these extensive data cleaning operations, we ensured the resulting dataset is reliable, representative, and ready for further preprocessing and modeling tasks.
rate.


## Data Preparation for Modeling

The predictive modeling process begins by segregating the dataset into features and labels. The labels represent four different binary delinquency outcomes, which provide details on whether a client is delinquent and, if so, whether the delinquency occurred over the course of 3, 6, or 12 months. 

The features exclude the IDs, all the above-mentioned labels, and a couple of other features such as `number_of_delinquent_months` and `average_delinquency_rate`. These dropped features would not be available at the time of prediction in a real-world scenario.

The data preparation process entails the following steps:

1. **Features and Labels Definition**: We first define our features and labels. Our features exclude IDs and various delinquency-related columns, ensuring we only include data available at the time of prediction in real-world situations. Our labels are the different binary delinquency outcomes.

2. **Train-Test Split**: We split our data into training and validation sets for each of the four delinquency outcomes using the `train_test_split` function from sklearn. We used an 80-20 split, with 80% of the data forming the training set and the remaining 20% forming the validation set. The random state was set to 42 to ensure repeatability of results.

3. **Feature Type Identification**: The features are further categorized into categorical and numerical features based on their data types. This step is essential for the subsequent feature encoding and scaling processes.

4. **Feature Encoding and Scaling**: For the categorical features, we used One-Hot Encoding (OHE), dropping the first category to avoid the dummy variable trap. For the numerical features, we applied Standard Scaling to standardize the feature values.

5. **Column Transformation**: We used the `ColumnTransformer` from sklearn to apply the above transformations to the appropriate features. This process ensures that our data is in a format suitable for machine learning algorithms.

After these steps, the data is prepared and ready for the modeling stage.

## Model Training and Scoring

We built a custom function `fit_and_score` to streamline the process of fitting our model to the data and evaluating its performance. This function accepts a GridSearchCV instance, the training and validation data for the features and labels, and a name string for identification purposes.

The function performs the following tasks:

1. **Model Fitting**: The function fits the GridSearchCV instance (our model along with hyperparameters to tune) on the training data.

2. **Training Score Calculation**: The function calculates the model's performance score on the training data.

3. **Testing Score Calculation**: The function also calculates the model's performance score on the validation data. This gives us an idea of how well our model generalizes to unseen data.

4. **Best Parameters Identification**: After fitting the model, the function identifies the best hyperparameters, providing insights into which parameters yield the best model performance.

5. **Storing Best Parameters**: The best parameters for each model are then returned by the function for later use and are stored in a dictionary.

By creating and utilizing the `fit_and_score` function, we can efficiently train and evaluate our models in a consistent manner. This approach helps ensure comparability of results across different models and target variables. It also aids in maintaining clean and readable code, as repetitive tasks are wrapped up in this function. 

We apply this function across all our target variables ("is_delinquent", "3mo_delinquency", "6mo_delinquency"), allowing us to fine-tune and evaluate models for predicting each of these outcomes.

## Model Implementation

Multiple machine learning models were used in this project, each with their unique capabilities to handle classification tasks. The models include Gradient Boosting, AdaBoost, Support Vector Classifier (SVC), Logistic Regression, Random Forest, and a Deep Neural Network. Below is a brief explanation for each:

1. **Gradient Boosting**: This model was initially run with default parameters, and then a grid search was performed to find the optimal hyperparameters. The parameter grid included varying learning rates, numbers of estimators, and max depth values.

2. **AdaBoost**: This model was also implemented with a decision tree classifier as a base estimator. A grid search was performed to optimize the learning rate, number of estimators, max depth of the base estimator, and the max features of the base estimator.

3. **Support Vector Classifier (SVC)**: This model was run initially with a radial basis function kernel and then optimized using our `fit_and_score` function.

4. **Logistic Regression**: A logistic regression model was also trained. A grid search was used to optimize the inverse of regularization strength (C), the penalty (l2 was used), the solver, and the maximum number of iterations.

5. **Random Forest**: A Random Forest Classifier was implemented, with a grid search to optimize the number of estimators, minimum samples required to split an internal node, the minimum number of samples required to be at a leaf node, and the number of features to consider when looking for the best split.

6. **Deep Neural Network (DNN)**: Lastly, a Deep Neural Network was built using the Keras library with TensorFlow as the backend. This model includes multiple layers of neurons with activation functions. It is compiled with binary cross-entropy as the loss function, Adam as the optimizer, and accuracy and recall as the metrics. The model also includes an early stopping mechanism to prevent overfitting by restoring the model weights from the epoch with the best value of the monitored quantity.

All models are trained across all our target variables ("is_delinquent", "3mo_delinquency", "6mo_delinquency"), with each model's best parameters stored in a dictionary for later use.

These models, collectively, provide a robust suite of predictive tools. Each model has its strengths and weaknesses, and using them together allows for better performance and the ability to capture a wider range of patterns in the data.

<<<<<<< HEAD

## The Winning Model: RandomForestClassifier

Random forests are an ensemble learning technique that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes or the mean prediction of the individual trees. RandomForestClassifier leverages the power of multiple decision-making models and creates a balanced and reliable prediction model that improves overall accuracy and reduces overfitting.

In our case, we employed the RandomForestClassifier in conjunction with the GridSearchCV for hyperparameter tuning. This allowed us to experiment with a set of potential parameter values defined in the param_grid. We trained this model on different prediction tasks such as `is_delinquent`, `3mo_delinquency`, and `6mo_delinquency`. 

## Training and Validation Results

For each of these prediction tasks, the model performance was:

1. **Is Delinquent:** 
   - Training Score: 86.97%
   - Testing Score: 78.78%
   - Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 400}

2. **3 Month Delinquency:**
   - Training Score: 85.78%
   - Testing Score: 79.33%
   - Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 300}

3. **6 Month Delinquency:**
   - Training Score: 85.66%
   - Testing Score: 79.05%
   - Best Parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 400}

The results illustrate a strong performance across all tasks, indicating the model's capacity to predict credit delinquency with a reasonable degree of accuracy.

## Model Persistence

The trained model was serialized to disk using pickle, a Python utility for object serialization. This approach allows for later reloading of the model for prediction without the need to retrain it. 

We saved the trained model to a file named `random_forest_model.pkl` in the working directory. This pickled model encapsulates the best parameters found during grid search and the state of the RandomForestClassifier that produced the best results. Thus, the model can be readily deployed in future predictive tasks, providing a fast and efficient solution.

In the future, this model can be further refined and its performance improved by training on more data, further tuning hyperparameters, or employing more complex ensemble methods.

=======

# Best Model: AdaBoost

Our project utilizes advanced machine learning techniques to help small and medium-sized banks predict customer delinquency. The current best model is an AdaBoost Classifier, which combines the strengths of multiple weak learners to form a strong predictive model.

## Model Parameters and Training

The AdaBoost model we used has a Decision Tree Classifier as its base estimator. We used GridSearchCV to find the best parameters for this model from a predefined parameter grid. Here are the parameters we used for the GridSearch:

- 'learning_rate': [2.25, 2.5, 2.7],
- 'n_estimators': [250, 300, 350],
- 'estimator__max_depth': [71, 81, 91, 101, None],
- 'estimator__max_features': ['auto', 'sqrt', 'log2']

## Results

The best AdaBoost model had the following parameters:

- 'n_estimators': 300
- 'learning_rate': 2.25
- 'estimator__max_depth': None
- 'estimator__max_features': 'auto'

This model achieved a training score of 0.9087652929813265 and a testing score of 0.7933032839665164, indicating a good balance between bias and variance and suggesting that the model generalizes well to unseen data. 

## Implications

The AdaBoost model's strength lies in its ability to focus on instances that are hard to classify, by assigning them higher weights in subsequent iterations. This characteristic makes it well-suited to the task of predicting customer delinquency, which often involves dealing with imbalanced datasets where delinquent customers may be in the minority.

By accurately predicting customer delinquency, our model enables banks to make more informed decisions when onboarding new customers, thus minimizing the risk of default. This will be particularly beneficial for small and medium-sized banks looking to optimize their risk assessment processes. 

## Saving and Loading the Model

We saved the trained model using the `pickle` module in Python. This allows us to easily reuse the model in the future without having to retrain it. The pickled model can be loaded and used to make predictions on new data with just a few lines of code. 

## Future Directions

While the AdaBoost model has proven effective, we're continually looking to refine our model. Future work will focus on exploring other machine learning techniques, fine-tuning model parameters, and incorporating additional relevant features to improve the model's predictive performance.
>>>>>>> f70595c9f245babcb19fe5375638997393489e89

## Project Contributors

This project has been a collaborative effort by our team members:

- Argishti Ovsepyan
- Masood Dastan
- Gabriela Fichtner
- Saamir Shamsie
