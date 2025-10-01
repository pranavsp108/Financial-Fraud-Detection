# Financial Fraud Detection using XGBoost


This repository contains a comprehensive machine learning project focused on detecting fraudulent financial transactions. Using a rich dataset of transaction records, customer information, and card data, this project builds and optimizes a robust fraud detection model with XGBoost. The notebook addresses common challenges in fraud detection, such as severe class imbalance, feature engineering from complex data types, and hyperparameter tuning for optimal performance.

---

## 💾 Dataset

The dataset used for this project is a synthetically generated financial dataset that combines transaction details, user demographics, card information, and merchant categories. Due to its size (~2 GB), it is not included in this repository.

* **Download Link:** [Financial Transactions Dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions](https://www.kaggle.com/code/pranavpadmannavar/financial-fraud-detection-using-xgboost/input))

**Setup Instructions:**
1.  Download the data from the link above.
2.  Create a folder named `data` in the root of this project.
3.  Unzip and place all `.csv` and `.json` files inside the `data/` folder.
4.  Ensure your notebook environment points to this directory to load the data.

---

## 🛠️ Tools and Libraries

This project leverages the following tools and libraries:

* **Programming Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook (via Kaggle)

---

## 📖 Project Methodology

The project follows a structured machine learning workflow, detailed in the notebook:

1.  **Data Loading & Merging:** Consolidated five separate data sources (`transactions`, `cards`, `users`, `mcc_codes`, `fraud_labels`) into a single, comprehensive DataFrame.

2.  **Data Preprocessing:**
    * Cleaned numerical columns by removing symbols (e.g., '$') and converting data types.
    * Handled missing `is_fraud` labels and converted the target variable to a binary format (0/1).
    * Performed stratified splitting of the data into training (60%), validation (20%), and test (20%) sets to maintain the class distribution.

3.  **Feature Engineering:**
    * **Date Engineering:** Extracted features from `datetime` objects, such as hour, day of the week, month, and days to card expiry.
    * **Cyclical Features:** Created sine and cosine transformations for time-based features to represent their cyclical nature (e.g., `hour_sin`, `day_of_week_cos`).
    * **Categorical Encoding:** Applied One-Hot Encoding to categorical variables. For high-cardinality features like `merchant_state`, less frequent categories were grouped into an 'OTHER' category to reduce dimensionality.
    * **Interaction Terms:** Created new features by multiplying high-importance features (e.g., `amount_x_online_transaction`) to capture synergistic effects.

4.  **Modeling & Evaluation:**
    * **Model Selection:** Trained two powerful gradient boosting models: **XGBoost** and **LightGBM**.
    * **Imbalance Handling:** Utilized the `scale_pos_weight` parameter, calculated as the ratio of negative to positive samples, to effectively manage the severe class imbalance.
    * **Evaluation Metrics:** Assessed model performance using metrics suitable for imbalanced datasets, including **ROC AUC Score**, **Precision**, **Recall**, and the **F1-Score**.

5.  **Hyperparameter Tuning:**
    * Performed `RandomizedSearchCV` on a sample of the training data to find the optimal hyperparameters for the XGBoost model, with `roc_auc` as the scoring metric.
    * Explored the impact of different classification thresholds (e.g., 0.15, 0.70) to analyze the trade-off between precision and recall.

---

## 📊 Key Results

The final, tuned XGBoost model demonstrated excellent performance on the unseen test data:

* **ROC AUC Score:** **0.9932**
* **Recall (Fraud Class):** **0.86** (Correctly identified 86% of all fraudulent transactions)
* **Precision (Fraud Class):** **0.33**

The high ROC AUC score indicates a strong ability to distinguish between classes, while the high recall is critical for minimizing missed fraud cases.

### Top Predictive Features

The model identified the following features as most important for detecting fraud:
* `amount_x_tolls` (Interaction Term)
* `description_Tolls and Bridge Fees`
* `use_chip_Online Transaction`
* `merchant_state_Italy`
* `description_Taxicabs and Limousines`

![Feature Importance Plot]

---

## 🚀 Future Work

* **Threshold Optimization:** Implement a detailed analysis (e.g., using Precision-Recall curves) to select an optimal classification threshold that balances the business costs of false positives and false negatives.
* **Advanced Feature Engineering:** Explore velocity features (e.g., transaction frequency over different time windows) and more complex geospatial features.
* **Alternative Models:** Experiment with other models like CatBoost or deep learning approaches designed for tabular data.
* **Anomaly Detection:** Integrate unsupervised methods to identify novel fraud patterns not seen in the training data.
