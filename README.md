# Spam Bot Detection

This project applies machine learning techniques to detect spam accounts based on features extracted from user profile data. Using logistic regression and random forest classifiers, the model achieved 94% precision and an 89% F1-score.

## Features Used
- Presence of a profile picture
- Ratio of numbers to letters in username
- Length of biography
- External URL presence
- Number of followers, following, and posts

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib (optional for visualization)

## Results
The random forest model provided high performance in identifying spam bots, making this approach suitable for lightweight social media moderation tasks.

## How to Run
1. Clone this repository
2. Navigate to the project directory
3. Run `spam_model.py` to train and test the model

## Dataset
This project uses synthetic data generated to reflect common spam indicators. Replace `spam_accounts.csv` with your own structured data for real-world applications.
