import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import learning_curve
import plotly.figure_factory as ff

# Load data
df = pd.read_csv('C:\\Users\\Home\\MachineLearning\\test_activities.csv')

# Define activity mappings to broader categories
activity_mapping = {
    'assemble a model': 'Crafting Activities',
    'assembling a simple puzzle': 'Recreational Games and Play',
    'attending a childrens concert': 'Music and Dance Activities',
    'attending a quiet matinee movie': 'Social and Community Activities',
    'bake cookies or a simple cake': 'Crafting Activities',
    'baking bread or pastries': 'Crafting Activities',
    'bike on a challenging path': 'Physical Activities',
    'bird watching': 'Outdoor and Nature Activities',
    'build with legos': 'Crafting Activities',
    'building a fort with blankets': 'Crafting Activities',
    'climb safely': 'Physical Activities',
    'collect and paint rocks': 'Artistic Activities',
    # Continue with your mappings...
}

# Apply mapping to consolidate activities
df['Activity_Category'] = df['Activity'].apply(lambda x: activity_mapping.get(x, 'Other'))

# Encode the target variable
label_encoder = LabelEncoder()
df['Activity_Encoded'] = label_encoder.fit_transform(df['Activity_Category'])

# Prepare features and target
X = df.drop(['Activity', 'Activity_Category', 'Activity_Encoded'], axis=1)
y = df['Activity_Encoded']

# One-hot encode the 'Emotion' column and prepare features
encoder = OneHotEncoder()
encoded_emotion = encoder.fit_transform(df[['Emotion']]).toarray()
emotion_features = encoder.get_feature_names_out(['Emotion'])
numeric_features = X.select_dtypes(include=[np.number]).values

# Combine encoded and numeric features
X_combined = np.hstack([encoded_emotion, numeric_features])

# Determine the smallest class size
min_class_count = y.value_counts().min()

# Apply SMOTE or Random Oversampling
if min_class_count > 1:
    k_neighbors = min(5, min_class_count - 1)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)
else:
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_combined, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


# Build a pipeline for scaling and classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)  # Correct placement of y_pred_prob generation

# Model evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# ROC curve calculation
y_test_bin = label_binarize(y_test, classes=np.unique(y_resampled))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(np.unique(y_resampled))):
    if np.sum(y_test_bin[:, i]) == 0:  # No positive samples
        continue
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue'])
for i, color in zip(range(len(np.unique(y_resampled))), colors):
    if i in roc_auc:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix visualization
# Assuming you've already computed 'y_test' and 'y_pred'
cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.inverse_transform(np.unique(y_resampled))  # This gets the unique labels as an array

# Correcting the labels conversion for Plotly
x_labels = labels.tolist()  # Convert numpy array to list for x-axis labels
y_labels = labels.tolist()  # Convert numpy array to list for y-axis labels

# Creating the confusion matrix heatmap with Plotly
fig = ff.create_annotated_heatmap(z=cm, x=x_labels, y=y_labels, colorscale='Blues', showscale=True)
fig.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label', tickmode='array', tickvals=list(range(len(x_labels))), ticktext=x_labels),
                  yaxis=dict(title='True label', tickmode='array', tickvals=list(range(len(y_labels))), ticktext=y_labels))
fig.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curve
plot_learning_curve(pipeline, "Learning Curve", X_resampled, y_resampled, cv=5, n_jobs=-1)
plt.show()
