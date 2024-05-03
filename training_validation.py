import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN

# Step 1: Data Preparation
df = pd.read_csv('test_activities.csv')

# Data Inspection
print("Columns:\n", df.columns)
print("Data Head:\n", df.head())

# Ensure 'Emotion' exists and is used properly
if 'Emotion' not in df.columns:
    raise ValueError("'Emotion' column is missing from the dataset.")

# Encode activity labels
label_encoder = LabelEncoder()
df['Activity'] = label_encoder.fit_transform(df['Activity'])

# Feature Engineering: Ensure 'Emotion' is present throughout
df['Rating_Emotion'] = df['Rating'].astype(str) + "_" + df['Emotion']

# One-hot encode this new interaction
df = pd.get_dummies(df, columns=['Emotion', 'Rating_Emotion'])

# Further processing, like class filtering and augmentation:
min_samples = 2  # Adjust if needed
df = df[df['Activity'].map(df['Activity'].value_counts()) >= min_samples]

# Split features and target
X = df.drop(columns=['Activity'])
y = df['Activity']

# Data Augmentation
smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution
print("New Class Distribution:\n", pd.Series(y_resampled).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Pipeline setup
pipeline = Pipeline([
    ('model', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model selection
best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=label_encoder.inverse_transform(np.unique(y_test)), yticklabels=label_encoder.inverse_transform(np.unique(y_test)))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Recommendation Function
def recommend_activity(user_rating, user_emotion, emotions):
    # Prepare user input
    user_input = pd.DataFrame(columns=X_train.columns)
    user_input['Rating'] = [user_rating]

    # One-hot encode emotions and interactions
    emotion_df = pd.DataFrame(0, index=[0], columns=['Emotion_' + e for e in emotions])
    interaction_df = pd.DataFrame(0, index=[0], columns=['Rating_Emotion_' + str(user_rating) + "_" + e for e in emotions])

    emotion_df['Emotion_' + user_emotion] = 1
    interaction_df['Rating_Emotion_' + str(user_rating) + "_" + user_emotion] = 1

    # Combine all inputs
    user_input = pd.concat([user_input, emotion_df, interaction_df], axis=1)

    # Drop duplicates
    user_input = user_input.loc[:, ~user_input.columns.duplicated()]

    # Reindex to match training order
    user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

    # Predicting Top Activities
    proba = best_model.predict_proba(user_input)[0]
    top_activities = np.argsort(-proba)[:3]
    return label_encoder.inverse_transform(top_activities)

# Test Recommendation
user_rating = 8  # Example rating
user_emotion = 'happy'  # Example emotion
emotions = ['happy', 'sad', 'angry', 'fear']

recommended_activities = recommend_activity(user_rating, user_emotion, emotions)

print("\nTop 3 Recommended Activities:")
for i, act in enumerate(recommended_activities, 1):
    print(f"{i}: {act}")
