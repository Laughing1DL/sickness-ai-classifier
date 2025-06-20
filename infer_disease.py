import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("model_sickness.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
symptoms = df.columns.tolist()
symptoms.remove('diseases')


def input_sickness(list_symptoms, user_input):
    text = user_input.lower()
    vector = []
    for symptom in list_symptoms:
        if symptom in text:
            vector.append(1)
        else:
            vector.append(0)
    return vector

user_input = input("What do you feel?: ")
vector = input_sickness(symptoms ,user_input)
vector = np.array(vector).reshape(1, -1)

prediction = model.predict(vector)
pred_idx = np.argmax(prediction)
sickness = label_encoder.inverse_transform([pred_idx])
print(f"Your condition might be: {sickness}")