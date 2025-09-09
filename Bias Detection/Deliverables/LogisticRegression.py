from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.preprocessing import LabelEncoder



article_df = pd.read_csv('/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/Main_Dataset1.csv')

le = LabelEncoder()
article_df['MATCH_LABELS_ENCODED'] = le.fit_transform(article_df['MATCH_LABELS'])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))


# 1. Prepare your texts as you did
texts = (
    article_df['cleaned_content'].fillna('') + " "
    + article_df['entities_Group'].fillna('').astype(str) + " "
    + article_df['Actions'].fillna('').astype(str) + " "
    + article_df['Key_Phrases'].fillna('').astype(str)
).tolist()
labels = article_df['MATCH_LABELS_ENCODED'].tolist()

# 2. Train-test split
train_data, val_data, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. Create and train a pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(train_data, train_labels)

# 4. Evaluate
preds = pipeline.predict(val_data)
print("Accuracy:", accuracy_score(val_labels, preds))
print(classification_report(val_labels, preds, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(val_labels, preds))

# 5. Save the model if needed
import os
import joblib
model_path = os.path.join(os.path.expanduser("~"), "Desktop/Dissertation/Bias Detection/logistic_regression_model.pkl")
joblib.dump(pipeline, model_path)

