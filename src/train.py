from src.data_loader import load_data
from src.model_builder import build_model
import tensorflow as tf

def train_model():
    train_dir = 'data/modified-dataset/train'
    test_dir = 'data/modified-dataset/test'

    train_data, val_data, _ = load_data(train_dir, test_dir)
    model = build_model(input_shape=(224, 224, 3), num_classes=train_data.num_classes)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save('models/e_waste_model.h5')

if __name__ == '__main__':
    train_model()
