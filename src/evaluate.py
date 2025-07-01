from src.data_loader import load_data
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model():
    test_dir = 'data/modified-dataset/test'
    _, _, test_data = load_data('data/modified-dataset/train', test_dir)

    model = tf.keras.models.load_model('models/e_waste_model.h5')
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

if __name__ == '__main__':
    evaluate_model()
