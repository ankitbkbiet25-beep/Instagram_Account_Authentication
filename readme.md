# Instagram Authenticity Detector: A High-Fidelity System to Identify Fake Accounts

## Project Goal

This project was built to address a critical challenge in platform integrity: reliably and automatically distinguishing between **genuine Instagram users** and **inauthentic profiles (bots or fakes)**. Our final model achieves high accuracy by looking beyond surface-level data, focusing instead on subtle, behavioral indicators.

The deliverable is a high-performing **Ensemble Model** deployed in a **Streamlit dashboard** for instant, actionable risk assessment.

***

## The Technical Advantage: Custom Feature Engineering

The key to our model's success wasn't just the algorithm; it was the data preparation. We found that raw profile metrics were not enough.

### The SuspicionScore

* **Innovation:** We engineered a composite metric called the **`SuspicionScore`** within the `model_resources/preprocessing.py` module.
* **What it does:** This score aggregates multiple low-level risk factors (e.g., zero posts, low engagement ratios, missing profile details) into a single, highly predictive numerical value. This amplified the subtle signals of deception, which is a common challenge in high-stakes classification.

### The Final Model: A Robust Ensemble

After rigorous testing (documented in `notebooks/insta.ipynb`), we finalized an **Ensemble Voting Classifier** (Soft Voting) in the `notebooks/insta_model.ipynb`. Combining the predictive power of several top-performing models ensures that our predictions are not only accurate but also **stable** and **generalizable** against new, unseen fake accounts.

***

## Key Performance Metrics

The model demonstrates exceptional capability in identifying the critical minority class (Fake Accounts), which is essential in a classification project of this nature:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.94** | The overall correctness of the system's predictions. |
| **F1-Score (Fake Class)** | **0.94** | Indicates a superb balance between **Precision** (avoiding false positives) and **Recall** (catching all fake accounts). This is the metric that matters most for risk detection. |

*Reference: `model_metrics.csv`*

***

## Repository Structure and Execution

This repository is organized to showcase the complete data science pipeline, from exploratory work to the final deployed product.

| File/Folder | Purpose |
| :--- | :--- |
| **`main.py`** | **The Deployed App.** Runs the Streamlit dashboard for real-time predictions. |
| **`notebooks/insta.ipynb`** | **Exploratory Phase.** Contains the initial EDA, feature engineering, and the comparative evaluation/GridSearch of candidate models. |
| **`notebooks/insta_model.ipynb`** | **Production Phase.** Focuses on training the final Ensemble Classifier, saving the full pipeline, and generating final metrics. |
| **`model_resources/`** | **The Production Assets.** Stores the saved `insta_voting_model.pkl`, the `power_transformer.pkl` for scaling, and the `preprocessing.py` logic. |
| **`BI Dashboard/`** | **Data Visualization.** Contains the Power BI dashboard (`insta.pbix`) for a visual exploration of the dataset. |
| **`Insta_train.csv` / `Insta_test.csv`** | **Data.** The necessary datasets for reproducibility. |

### How to Run the Live Dashboard

1.  **Install requirements:** Make sure all libraries (streamlit, pandas, joblib, sklearn, etc.) are installed.
2.  **Execute:** Run the application from your terminal:
    ```bash
    streamlit run main.py
    ```