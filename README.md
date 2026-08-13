# Mobile-Product-Segmentation-and-Recommendation-System-
Mobile Product Segmentation and Recommendation System Using Python and Machine Learning
# 🤖 E-Commerce Mobile Product Recommendation System

An interactive **E-Commerce Mobile Product Recommendation System** built using **Python, Machine Learning, Pandas, Scikit-learn, Plotly, and Streamlit**.

The project analyzes mobile product data, handles missing/null values, performs **unsupervised machine learning using clustering**, and recommends similar mobile phones using **cosine similarity**.

---

## 📌 Project Overview

The goal of this project is to build an intelligent recommendation system for mobile products.

The system performs two major tasks:

1. **Data Analysis**

   * Explore mobile product information.
   * Analyze brands and models across clusters.
   * Compare product ratings between clusters.
   * Identify low design-rating products.
   * Analyze battery-life ratings.
   * Analyze product sources.

2. **Recommendation System**

   * Select a reference mobile phone.
   * Identify its machine-learning cluster.
   * Filter products based on brand, rating, and price.
   * Calculate similarity using cosine similarity.
   * Display the top recommended mobile phones with match scores.

---

## 🎯 Objectives

* Clean and preprocess the raw e-commerce dataset.
* Handle missing/null values.
* Prepare numerical and categorical features for machine learning.
* Discover groups of similar products using unsupervised learning.
* Build an effective product recommendation system.
* Provide an interactive dashboard using Streamlit.
* Visualize product and cluster-level insights using Plotly.

---

## 🧠 Machine Learning Approach

This project uses **unsupervised machine learning** because the dataset does not require a predefined target variable for clustering.

### Unsupervised Learning

The product dataset is grouped into clusters based on product characteristics such as:

* Performance rating
* Camera rating
* Battery-life rating
* Design rating
* Helpful votes
* Other processed product features

The clustering algorithm identifies products with similar characteristics.

The resulting cluster is stored in the dataset as:

```text
cluster
```

The recommendation system then uses this cluster to reduce the search space before calculating product similarity.

---

## 🔄 Project Workflow

```text
Raw E-Commerce Dataset
          ↓
      Data Cleaning
          ↓
   Handle Null Values
          ↓
 Feature Preprocessing
          ↓
 Unsupervised Learning
      (Clustering)
          ↓
   Processed Dataset
          ↓
 Cosine Similarity
          ↓
 Recommendation System
          ↓
    Streamlit Dashboard
```

---

## 🧹 Data Cleaning

Before applying machine learning, the dataset was cleaned and processed.

The preprocessing stage includes:

* Checking missing values.
* Handling null values.
* Removing or treating invalid records.
* Converting columns into appropriate data types.
* Preparing numerical features.
* Encoding categorical features where required.
* Scaling/transforming features for machine learning.
* Creating a processed feature matrix.

The cleaned dataset is saved as:

```text
processed_df.pkl
```

The processed machine-learning feature matrix is saved as:

```text
x_processed.pkl
```

---

## 📊 Data Analysis Dashboard

The Streamlit application contains a **Data Analysis** page with multiple visualizations.

### 1. Brand Distribution Across Clusters

Shows how different brands are distributed across the machine-learning clusters.

### 2. Model Counts by Cluster

Displays the number of models belonging to each cluster.

### 3. Cluster-Wise Rating Comparison

Compares average:

* Performance rating
* Camera rating
* Design rating
* Helpful votes

between different clusters.

### 4. Design Rating Analysis

Products with low design ratings are identified using:

```python
df[df["design_rating"] < 2.5]
```

A box plot is used to understand the distribution and potential outliers.

### 5. Battery-Life Analysis

A violin plot is used to visualize battery-life rating distributions across clusters.

### 6. Source Distribution

A cross-tabulation and bar chart are used to analyze the distribution of products across different sources.

---

## 🤖 Recommendation System

The recommendation system uses a combination of:

### 1. Cluster-Based Filtering

When a user selects a mobile phone, its cluster is identified:

```python
target_cluster = df.loc[selected_index, "cluster"]
```

Only products belonging to the same cluster are considered initially.

This helps recommend products with similar characteristics.

---

### 2. Dynamic Filtering

Users can apply the following filters:

#### 🏷️ Brand

Select one or more brands.

#### ⭐ Minimum Rating

Choose the minimum acceptable product rating.

#### 💰 Maximum Price

Set the maximum product price.

#### 🔢 Number of Recommendations

Choose between 1 and 10 recommendations.

---

### 3. Cosine Similarity

After filtering, the system calculates similarity between the selected mobile and candidate products.

```python
sim_scores = cosine_similarity(
    target_vector,
    cluster_vectors
)[0]
```

Products with higher cosine similarity are considered more similar to the selected mobile.

The recommendations are sorted according to similarity score.

---

## 📈 Match Score

Each recommended product displays a similarity percentage.

For example:

```text
Match Score: 94.5%
```

The score is calculated from the cosine similarity value:

```python
score * 100
```

A higher score indicates that the product has a more similar feature representation to the selected mobile.

---

## 🖥️ Streamlit Application

The application contains two pages:

```text
Menu
├── Data Analysis
└── Recommendation System
```

### Data Analysis

Provides interactive charts and tables for exploring the dataset.

### Recommendation System

Allows users to:

1. Select a mobile phone.
2. Select preferred brands.
3. Set a minimum rating.
4. Set a maximum price.
5. Select the number of recommendations.
6. View recommended products.
7. View the similarity match score.

---

## 🛠️ Technologies Used

| Technology     | Purpose                                |
| -------------- | -------------------------------------- |
| Python         | Main programming language              |
| Pandas         | Data manipulation and analysis         |
| NumPy          | Numerical operations                   |
| Scikit-learn   | Machine learning and cosine similarity |
| Plotly Express | Interactive visualizations             |
| Streamlit      | Web application/dashboard              |
| Pickle         | Saving and loading processed ML data   |

---

## 📂 Project Structure

```text
E-Commerce-Recommendation-System/
│
├── app.py
│
├── processed_df.pkl
├── x_processed.pkl
│
├── requirements.txt
│
├── README.md
│
└── data/
    └── raw_dataset.csv
```

> `processed_df.pkl` contains the cleaned/processed DataFrame, while `x_processed.pkl` contains the processed feature matrix used for similarity calculations.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/e-commerce-recommendation-system.git
```

### 2. Navigate to the Project Directory

```bash
cd e-commerce-recommendation-system
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

---

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔍 Example Recommendation Workflow

Suppose the user selects:

```text
Reference Mobile: Samsung Galaxy S23
Brand: Samsung
Minimum Rating: 4.0
Maximum Price: 50000
Top N: 5
```

The system will:

```text
Samsung Galaxy S23
        ↓
Find Product Cluster
        ↓
Filter Same Cluster
        ↓
Apply Brand Filter
        ↓
Apply Rating Filter
        ↓
Apply Price Filter
        ↓
Calculate Cosine Similarity
        ↓
Sort Similarity Scores
        ↓
Return Top 5 Products
```

---

## 💡 Advantages

* Interactive user interface.
* Uses unsupervised machine learning.
* Handles missing/null data during preprocessing.
* Supports multiple recommendation filters.
* Uses cosine similarity for product matching.
* Provides visual data analysis.
* Displays recommendation match scores.
* Easy to extend with additional product features.

---

## 🚀 Future Improvements

The project can be improved by adding:

* Personalized recommendations based on user history.
* Collaborative filtering.
* Hybrid recommendation techniques.
* Product images.
* Product review sentiment analysis.
* Price comparison across different sources.
* Real-time product data.
* User login and recommendation history.
* Better recommendation ranking.
* Deployment using Streamlit Cloud or another cloud platform.
* Explainable recommendations showing *why* a product was recommended.

---

## 📌 Important Notes

The recommendation quality depends on the quality of the processed features stored in `x_processed.pkl`.

The system currently uses:

```text
Cluster → Filter → Cosine Similarity → Top N
```

Therefore, improving feature engineering and clustering can directly improve recommendation quality.

---

## 👨‍💻 Project Summary

This project demonstrates how **data preprocessing, unsupervised machine learning, similarity-based recommendation, data visualization, and Streamlit application development** can be combined to create an end-to-end E-Commerce recommendation system.

The final application provides both **business-oriented data insights** and an **interactive product recommendation experience**.
