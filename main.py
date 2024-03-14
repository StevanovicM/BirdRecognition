# Data Science Biblioteke
from helper_functions import walk_through_dir, create_tensorboard_callback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Tensorflow Biblioteke
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# Sistemkse Biblioteke
from pathlib import Path
import os.path

# Biblioteke za vizualizaciju
import seaborn as sns

sns.set_style('darkgrid')

# Metrike
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Učitavamo i transformišemo podatke
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

dataset='archive/train'
walk_through_dir(dataset)

# Postavljanje podataka u DataFrame
# Prva kolona filepaths sadrži lokaciju putanje datoteke svake pojedinačne slike.
# Druga kolona labels, sadrži oznaku klase odgovarajuće slike sa putanje datoteke.

image_dir = Path(dataset)

# Skupljanje filepaths i labels variabli
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Konkatenacija filepaths i labels
image_df = pd.concat([filepaths, labels], axis=1)

# Preprocessing podataka
# Podaci će biti razdvojeni u 3 seta: Trening, Validacija i Testiranje.
# Podaci o obuci će se koristiti za obuku CNN modela dubokog učenja i njegovi parametri će biti fino podešeni sa podacima o validaciji.
# Konačno, performanse podataka će biti procenjene korišćenjem testnih podataka (podaci koje model ranije nije video).

# Deljenje podataka na train i test
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)

# Deljenje slika u 3 kategorije
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

# Korak augmentacije podataka
augment = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(224,224),
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.1),
  layers.experimental.preprocessing.RandomZoom(0.1),
  layers.experimental.preprocessing.RandomContrast(0.1),
])

# Treniranje modela
# Slike modela će biti podvrgnute unapred obučenom CNN modelu pod nazivom EfficientNetB0.
# Tri callbacka će se koristiti za praćenje obuke.
# To su: Model Checkpoint, Rano zaustavljanje, Tensorboard callback. Sažetak hiperparametra modela je prikazan na sledeći način:
# Veličina batcha: 32
# Epohe: 100
# Oblik inputa: (224, 224, 3)
# Output sloj: 525

# Učitavanje predobučenog CNN modela
print("Loading the CNN model")
pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)

pretrained_model.trainable = False

checkpoint_dir = "birds_classification_model_checkpoint"
checkpoint_model_path = os.path.join(checkpoint_dir, "birds_classification_model.h5")

if os.path.exists(checkpoint_model_path):
    print("Loading weights from the previous checkpoint")
    pretrained_model = load_model(checkpoint_model_path)

# Kreairanje checkpointa
checkpoint_path = "birds_classification_model_checkpoint"
checkpoint_callback = ModelCheckpoint(checkpoint_model_path,
                                      save_weights_only=False,
                                      monitor="val_accuracy",
                                      save_best_only=False)

# Podešavanje EarlyStopping callbacka da zaustavi trening ako se modelova val_loss metrika ne poboljša za 3 epohe
early_stopping = EarlyStopping(monitor="val_loss", # pratimo val loss metriku
                               patience=5,
                               restore_best_weights=True) # ako se val loss smanji za 3 epohe zaredom, zaustavi trening

# Smanjenje stope učenja ukoliko se metrike ne budu poboljšavale
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Treniranje modela
inputs = pretrained_model.input
x = augment(inputs)

x = Dense(128, activation='relu')(pretrained_model.output)
x = Dropout(0.45)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.45)(x)


outputs = Dense(525, activation='softmax')(x)

model = load_model(checkpoint_model_path)

model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=100,
    callbacks=[
        early_stopping,
        create_tensorboard_callback("training_logs",
                                    "bird_classification"),
        checkpoint_callback,
        reduce_lr
    ]
)

print("Evaluating model")

# Evaluacija modela
results = model.evaluate(test_images, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

# Predviđanje na test skupu podataka
y_pred_prob = model.predict(test_images)
y_pred = np.argmax(y_pred_prob, axis=1)

# Postavljanje pravih klasa
y_true = test_images.classes

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("EfficientNetB0 Model Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Vizualizacija loss krivih

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predviđanje klasa test slika
pred = model.predict(test_images)
pred = np.argmax(pred,axis=1)

# Mapiranje labela
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

# Prikazivanje rezultata
print(f'The first 5 predictions: {pred[:5]}')

# Prikazivanje nasumičnih 15 slika sa njihovim klasama
random_index = np.random.randint(0, len(test_df) - 1, 15)
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[random_index[i]]))
    if test_df.Label.iloc[random_index[i]] == pred[random_index[i]]:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {pred[random_index[i]]}", color=color)
plt.show()
plt.tight_layout()

y_test = list(test_df.Label)
print(classification_report(y_test, pred))
