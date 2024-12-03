from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def train_vgg16(dataset_path, img_height, img_width, batch_size, epochs, num_classes, model_save_path):
    # Data augmentation and rescaling
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Debug lines
    print("Class Indices: ", train_generator.class_indices)
    print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes.")

    # Load VGG16 base model
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in vgg_base.layers:
        layer.trainable = False

    # Add custom layers
    model = Sequential([
        vgg_base,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        verbose=1
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Plot Accuracy and Loss
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Model Training - VGG16')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    dataset_path = '/content/drive/MyDrive/dataset'
    model_save_path = '/content/drive/MyDrive/models/vgg16_family_member_model.h5'
    train_vgg16(dataset_path, img_height=128, img_width=128, batch_size=15, epochs=20, num_classes=3, model_save_path=model_save_path)
