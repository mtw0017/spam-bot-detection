# Spam Bot Detection

This project applies supervised machine learning techniques to detect fake Instagram accounts based on profile features. Models were evaluated for accuracy and learning progression to determine their effectiveness in classifying spam bots.

## Features Used

- Presence of a profile picture
- Ratio of numbers to letters in the username
- Number of words in the full name
- Ratio of numbers to words in the full name
- Whether the name matches the username
- Length of bio (description)
- External URL in bio
- Account privacy status
- Number of posts, followers, and accounts followed

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Seaborn, Matplotlib

## Models Evaluated

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine (SVM, SVC)
- AdaBoost
- Gradient Boosting
- XGBoost

## Visualizations

- **Heatmap** to check correlation between features
- **Pairplot** to observe feature distribution and clusters
- **Learning Curves** to visualize how model performance changes as training size increases

## Results

- **Random Forest and Decision Tree** achieved the highest accuracy.
- **Logistic Regression** achieved 94% precision and an 89% F1-score.
- **SVM** had the lowest performance, possibly due to lack of feature separation.
- Most spam accounts had shorter bios, fewer posts, and fewer followers, but correlations were not statistically strong enough to be considered reliable across all cases.

## How to Run

1. Clone this repository
2. Install dependencies:  
   ```bash
   pip install pandas scikit-learn xgboost seaborn matplotlib
