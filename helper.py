from keras.utils import image_dataset_from_directory
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.data.experimental import cardinality

RAW_DATASET_CACHE = ".cache/extracted/Rice_Image_Dataset"
SEED = 96

def get_data(image_size):
    (train_data, test_data) = image_dataset_from_directory(
        RAW_DATASET_CACHE,
        image_size=image_size,
        validation_split=0.3,
        subset="both",
        seed=SEED,
        verbose=False
    )

    label_names = train_data.class_names

    train_size = cardinality(train_data).numpy()
    test_size = cardinality(test_data).numpy()

    train_samples = train_data.shuffle(10000, seed=SEED).take(int(train_size * 0.25))
    validation_samples = test_data.shuffle(10000, seed=SEED).take(int(test_size * 0.25))

    return train_samples, validation_samples, label_names


def plot_samples(plt, train_images, label_names):
    plt.figure(figsize=(20, 6))

    for images, labels in train_images.take(1):
        for i in range(15):
            plt.subplot(3, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(label_names[labels[i]])
            plt.axis("off")
    plt.show()

def plot_accuracy(plt, accuracy, validation_accuracy):
    plt.figure(figsize=(8, 4))
    plt.plot(accuracy, label="training accuracy")
    plt.plot(validation_accuracy, label="validation accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def plot_loss(plt, loss, validation_loss):
    plt.figure(figsize=(8, 4))
    plt.plot(loss, label="training loss")
    plt.plot(validation_loss, label="validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, test_images, label_names):
    true = []
    pred = []

    # Numpy Iterator has better performance
    for images, labels in test_images.as_numpy_iterator():
        true.extend(labels)
        pred.extend(model.predict(images, verbose=0).argmax(axis=1))

    ConfusionMatrixDisplay.from_predictions(
        y_true=true,
        y_pred=pred,
        display_labels=label_names,
        normalize="true",
        cmap="gray"
    )
