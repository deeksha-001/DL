import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================
# CREATE MODELS FOLDER
# =====================================

os.makedirs("models", exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("data/heart.csv")

print("Dataset Shape:", df.shape)

# =====================================
# ENCODE CATEGORICAL COLUMNS
# =====================================

categorical_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =====================================
# FEATURE SCALING
# =====================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# BUILD ANN MODEL
# =====================================

model = Sequential()

model.add(
    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dropout(0.2))

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# EARLY STOPPING
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================
# EVALUATE MODEL
# =====================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# =====================================
# SAVE MODEL
# =====================================

model.save("models/ann_model.h5")

print("\nModel Saved Successfully!")
print("Scaler Saved Successfully!")
print("Encoders Saved Successfully!")