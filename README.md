//Sickness AI Classifier

A neural network-based disease predictor using free-text symptom input and a structured symptom-disease dataset.

//Features

- Accepts user symptom descriptions in natural language
- Converts text to a binary vector of symptom indicators
- Predicts the most likely disease using a trained neural network
- Built with TensorFlow, Keras, Pandas, and Scikit-learn

//File Structure

----data/
-------sample_dataset.csv # Small sample dataset (100 rows)
----modelo_enfermedades.h5 # Trained Keras model
----label_encoder.pkl # Encoder for class labels
----train_model.py # Model training script
----infer_disease.py # User interaction + prediction script
----README.md

//Example

Input: "I have fever, fatigue, and chest pain"
Output: "Your condition may be: Tuberculosis"

//Dataset

The full dataset is too large for GitHub.  
Instead, a 100-row sample is provided in `data/sample_dataset.csv` for demonstration purposes.

If you wish to train on the full dataset, structure your file with:
- Column: `diseases` (labels)
- Remaining columns: symptoms (binary 0/1 indicators)

//Requirements

- Python 3.10+
- pandas
- numpy
- tensorflow
- scikit-learn

To install dependencies:

```bash
pip install -r requirements.txt

Author:
Axel Mora — aspiring AI engineer using machine learning and deep learning.
