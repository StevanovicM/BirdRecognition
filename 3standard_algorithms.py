# Import Data Science Libraries
from helper_functions import walk_through_dir, create_tensorboard_callback
import keras.callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Tensorflow Libraries
from keras import layers, Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import keras_tuner as kt

# System libraries
from pathlib import Path
import os.path

# Visualization Libraries
import seaborn as sns

sns.set_style('darkgrid')

# Metrics
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Load and transform data
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

dataset='archive/train'
walk_through_dir(dataset)

# Placing data into a Dataframe
# The first column filepaths contains the file path location of each individual images.
# The second column labels, on the other hand, contains the class label of the corresponding image from the file path

image_dir = Path(dataset)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

# Visualizing images from the dataset
# Display 16 picture of the dataset with their labels
random_index = np.random.randint(0, len(image_df), 16)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()

# Data Preprocessing
#The data will be split into three different categories: Training, Validation and Testing.
# The training data will be used to train the deep learning CNN model and its parameters will be fine tuned with the validation data.
# Finally, the performance of the data will be evaluated using the test data(data the model has not previously seen).

# Separate in train and test data
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)

# Split the data into three categories.
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=TARGET_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Data Augmentation Step
augment = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(224,224),
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.1),
  layers.experimental.preprocessing.RandomZoom(0.1),
  layers.experimental.preprocessing.RandomContrast(0.1),
])

# Non-Deep learning algorithms
# Since non-deep learning models cannot directly handle the image data as fed into a CNN,
# we'll need to flatten the image arrays and potentially reduce their dimensionality due to their large size.
# This will make the computation tractable for these algorithms.

# Implementing dimensionality reduction using PCA (Principal Component Analysis)
# Before applying PCA, the images need to be processed into a form that PCA can work with:
# Filtering the dataframe for a subset of classes
selected_classes = image_df['Label'].value_counts().index[:20] # Select the top 20 classes
filtered_df = image_df[image_df['Label'].isin(selected_classes)]

filtered_image_list = [img_to_array(load_img(img_path, target_size=TARGET_SIZE))
                       for img_path in filtered_df['Filepath']]
filtered_image_array = np.array(filtered_image_list)

# Flatten the image data for PCA
X = filtered_image_array.reshape(filtered_image_array.shape[0], -1)

# Encode labels into numerical values
le = LabelEncoder()
y = le.fit_transform(filtered_df['Label'].values)

print("Printing image array")
print(filtered_image_array.shape)  # This should now be (number_of_images, TARGET_SIZE[0]*TARGET_SIZE[1]*3)

# Split the data into train and test sets
print("Splitting the data into training and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling using StandardScaler")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now we can apply PCA
print("Applying PCA")
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Instantiate the models
print("Instantiating the models")
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
logreg = LogisticRegression(max_iter=10000, random_state=42)

# Fit the models
print("Fitting the models")
knn.fit(X_train_pca, y_train)
rf.fit(X_train_pca, y_train)
logreg.fit(X_train_pca, y_train)

# Number of folds
k = 5

# Perform cross-validation for k-NN
cross_val_scores_knn = cross_val_score(knn, X_train_pca, y_train, cv=k, scoring='accuracy')
print(f"k-NN cross-validation accuracy scores: {cross_val_scores_knn}")
print(f"Mean k-NN cross-validation accuracy: {cross_val_scores_knn.mean()}")

# Perform cross-validation for Random Forest
cross_val_scores_rf = cross_val_score(rf, X_train_pca, y_train, cv=k, scoring='accuracy')
print(f"Random Forest cross-validation accuracy scores: {cross_val_scores_rf}")
print(f"Mean Random Forest cross-validation accuracy: {cross_val_scores_rf.mean()}")

# Perform cross-validation for Logistic Regression
cross_val_scores_logistic = cross_val_score(logreg, X_train_pca, y_train, cv=k, scoring='accuracy')
print(f"Logistic Regression cross-validation accuracy scores: {cross_val_scores_logistic}")
print(f"Mean Logistic Regression cross-validation accuracy: {cross_val_scores_logistic.mean()}")


# Tuning the hyperparameters

# For k-NN, we use GridSearchCV to find the best number of neighbors:
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_gs.fit(X_train_pca, y_train)

best_knn = knn_gs.best_estimator_

# For Random Forest, GridSearchCV can be used to tune parameters like the number of trees and depth of each tree:
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30, None]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_gs.fit(X_train_pca, y_train)

best_rf = rf_gs.best_estimator_

# For Logistic Regression, we tune the regularization strength and the type of penalty:
logistic_params = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
logistic_gs = GridSearchCV(LogisticRegression(max_iter=1000), logistic_params, cv=5)
logistic_gs.fit(X_train_pca, y_train)

best_logistic = logistic_gs.best_estimator_


# Cross-validation after hyperparameter tuning
cross_val_scores_knn = cross_val_score(best_knn, X_train_pca, y_train, cv=5, scoring='accuracy')
print(f"k-NN cross-validation accuracy scores after hyperparameter tuning: {cross_val_scores_knn}")
print(f"Mean k-NN cross-validation accuracy after hyperparameter tuning:: {cross_val_scores_knn.mean()}")

cross_val_scores_rf = cross_val_score(best_rf, X_train_pca, y_train, cv=k, scoring='accuracy')
print(f"Random Forest cross-validation accuracy scores after hyperparameter tuning: {cross_val_scores_rf}")
print(f"Mean Random Forest cross-validation accuracy after hyperparameter tuning: {cross_val_scores_rf.mean()}")

# Perform cross-validation for Logistic Regression
cross_val_scores_logistic = cross_val_score(best_logistic, X_train_pca, y_train, cv=k, scoring='accuracy')
print(f"Logistic Regression cross-validation accuracy scores after hyperparameter tuning: {cross_val_scores_logistic}")
print(f"Mean Logistic Regression cross-validation accuracy after hyperparameter tuning: {cross_val_scores_logistic.mean()}")

# Evaluating the models

# Predictions on the test set
y_pred_knn = best_knn.predict(X_test_pca)
y_pred_rf = best_rf.predict(X_test_pca)
y_pred_logistic = best_logistic.predict(X_test_pca)

# Evaluation reports for the traditional models
class_names = le.classes_
report_knn = classification_report(y_test, y_pred_knn, target_names=class_names)
report_rf = classification_report(y_test, y_pred_rf, target_names=class_names)
report_logistic = classification_report(y_test, y_pred_logistic, target_names=class_names)

print("k-NN Classification Report:\n", report_knn)
print("Random Forest Classification Report:\n", report_rf)
print("Logistic Regression Classification Report:\n", report_logistic)