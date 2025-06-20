import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
import re
import pickle

df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
X = df.drop('diseases', axis=1)
y = df['diseases']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dense(256, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
model.save("model_sickness.h5")
print("Model saved correctly")