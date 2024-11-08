# Amazon_product_reviews_SentimentAnalysis-and-TopicModeling
Project Overview
This project aims to perform sentiment analysis and topic modeling on a dataset of Amazon product reviews. The objective is to build a model that accurately predicts the sentiment of reviews (positive, neutral, or negative) and to identify key topics discussed by users. Using natural language processing (NLP) techniques and machine learning, this project provides insights into customer sentiment and common themes in the reviews.

Dataset
The dataset consists of Amazon product reviews, including the following columns:

Text: The full review text.
Summary: A brief summary of the review.
Score: A rating score from 1 to 5.
HelpfulnessNumerator and HelpfulnessDenominator: Representing the number of users who found the review helpful.
Time: The timestamp of the review.
Project Workflow
1. Data Preprocessing
Data Cleaning: Removed duplicates and handled missing values to ensure data quality.
Text Preprocessing:
Combined Summary and Text columns to create a comprehensive review text (Combined_Text).
Cleaned the text by removing punctuation, converting to lowercase, and removing stopwords.
Sentiment Labeling:
Mapped Score values to sentiment labels: scores of 4-5 as positive, 3 as neutral, and 1-2 as negative.
Class Balancing:
Balanced the dataset by downsampling the positive and negative classes to match the neutral class, ensuring an equal representation of sentiments.
2. Exploratory Data Analysis (EDA)
Sentiment Distribution: Visualized the sentiment distribution to confirm balanced classes.
Text Length Analysis: Analyzed the length of reviews to understand variations across sentiments.
Helpfulness Ratio Calculation: Created a ratio from HelpfulnessNumerator and HelpfulnessDenominator to identify helpful reviews and included it as a feature for model training.
3. Text Vectorization
Used TF-IDF (Term Frequency-Inverse Document Frequency) to convert Combined_Text into a matrix of features, setting max_features to 5000 to capture important terms without overfitting.
The TF-IDF matrix shape after vectorization was (62738, 5000).
4. Model Training
Logistic Regression:
Trained a Logistic Regression model on the TF-IDF features.
Model Evaluation: Achieved an accuracy of 75.1% with the following classification report:
Negative: Precision = 0.77, Recall = 0.81, F1-score = 0.79
Neutral: Precision = 0.66, Recall = 0.63, F1-score = 0.65
Positive: Precision = 0.81, Recall = 0.80, F1-score = 0.80
5. Advanced Model Exploration
Additionally trained Support Vector Machine (SVM) and Random Forest Classifier models as advanced methods to compare performance. SVM performed well, while Random Forest provided robust results, especially for detecting sentiment nuances in neutral reviews.
6. Topic Modeling
Latent Dirichlet Allocation (LDA):
Implemented LDA for topic modeling on the Combined_Text column.
Identified top keywords in each topic to understand common themes in the reviews.
Model Performance Visualization
To visualize model performance, the following charts were generated:

Confusion Matrix: Displayed true vs. predicted labels for each class, providing insight into misclassifications.
Classification Report Heatmap: Showed precision, recall, and F1-scores across each sentiment class.
ROC Curve: Illustrated the true positive rate vs. false positive rate for each sentiment class.
Conclusion
The project demonstrates a successful application of sentiment analysis and topic modeling on a large dataset of Amazon product reviews. The Logistic Regression model provided solid baseline results, while advanced models offered additional insights. Topic modeling revealed common themes across product reviews, giving valuable insights into customer perspectives.

Future Work
Future enhancements could include:

Experimenting with transformer-based models like BERT for improved sentiment accuracy.
Fine-tuning the LDA model for more refined topic extraction.
Adding additional features, such as review length or product category, to improve model performance.

