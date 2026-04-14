from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from helper import *
import keras_tuner as kt
import tensorflow as tf

image_size = (64, 64)
batch_size = 32

(all_images, train_images, validation_images, test_images, label_names) = get_data(image_size)


AUTOTUNE = tf.data.AUTOTUNE

train_images = (
    train_images
    .shuffle(2000)
    .batch(batch_size)
    .cache()
    .prefetch(AUTOTUNE)
)

validation_images = (
    validation_images
    .batch(batch_size)
    .cache()
    .prefetch(AUTOTUNE)
)

def create_model(hp):
    dropout = hp.Float(
        "dropout",
        min_value=0.1,
        max_value=0.5,
        step=0.05
    )

    l2_value = hp.Float(
        "l2",
        min_value=1e-5,
        max_value=1e-2,
        sampling="log"
    )

    learning_rate = hp.Float(
        "learning_rate",
        1e-5,
        1e-3,
        sampling="log"
    )

    filters1 = hp.Choice("filters1", [32, 64, 128])
    filters2 = hp.Choice("filters2", [32, 64, 128])
    filters3 = hp.Choice("filters3", [32, 64])

    regularizer = regularizers.l2(l2_value)

    new_model = models.Sequential([
        layers.Input(shape=image_size + (3,)),
        layers.Rescaling(1./255),

        layers.Conv2D(filters1, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(filters2, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(filters3, (3,3), activation="relu", kernel_regularizer=regularizer),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), activation="relu", kernel_regularizer=regularizer),

        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(len(label_names), activation="softmax", kernel_regularizer=regularizer, dtype="float32")
    ])
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return new_model

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

tuner = kt.Hyperband(
    create_model,
    objective="val_accuracy",
    max_epochs=50,   # ↓ faster
    factor=3,
    hyperband_iterations=2,  # KEY SPEED PARAM
    directory="tuning",
    project_name="cnn_fast"
)

tuner.search(
    train_images,
    validation_data=validation_images,
    epochs=30,
    callbacks=[early_stop]
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]
best_model.save("best_model.keras")

print("Best dropout:", best_hp.get("dropout"))
print("Best L2:", best_hp.get("l2"))
print("Best LR:", best_hp.get("learning_rate"))

tuner.results_summary()