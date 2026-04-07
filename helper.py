from keras.utils import image_dataset_from_directory
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from numpy import ceil, floor

RAW_DATASET_CACHE = ".cache/extracted/Rice_Image_Dataset"
SEED = 96
BATCH_SIZE = 32
NUM_OF_IMAGES = 20000
TRAIN_SPLIT = 0.7 * 0.8
VALIDATION_SPLIT = 0.7 * 0.2
TEST_SPLIT = 0.3

NUMBER_OF_BATCHES = NUM_OF_IMAGES // BATCH_SIZE

NUMBER_OF_TRAIN_BATCHES = ceil(NUMBER_OF_BATCHES * TRAIN_SPLIT) # ceil because of floating point precision error
NUMBER_OF_VALIDATION_BATCHES = floor(NUMBER_OF_BATCHES * VALIDATION_SPLIT)
NUMBER_OF_TEST_BATCHES = floor(NUMBER_OF_BATCHES * TEST_SPLIT)

def get_data(image_size):
    data = image_dataset_from_directory(
        RAW_DATASET_CACHE,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        verbose=False
    )

    label_names = data.class_names

    data = data.shuffle(10000, seed=SEED)

    train_samples = data.take(NUMBER_OF_TRAIN_BATCHES)
    validation_samples = data.skip(NUMBER_OF_TRAIN_BATCHES).take(NUMBER_OF_VALIDATION_BATCHES)
    test_samples = data.skip(NUMBER_OF_TRAIN_BATCHES + NUMBER_OF_VALIDATION_BATCHES).take(NUMBER_OF_TEST_BATCHES)

    return train_samples, validation_samples, test_samples, label_names


def plot_samples(train_images, label_names):
    plt.figure(figsize=(16, 4))

    for images, labels in train_images.take(1):
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(label_names[labels[i]])
            plt.axis("off")
    plt.show()

def plot_accuracy_and_loss(accuracy, validation_accuracy, loss, validation_loss):
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label="training accuracy")
    plt.plot(validation_accuracy, label="validation accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="training loss")
    plt.plot(validation_loss, label="validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")

    plt.legend()
    plt.show()

def calculate_predictions(model, test_images):
    true = []
    pred = []

    # Numpy Iterator has better performance
    for images, labels in test_images.as_numpy_iterator():
        true.extend(labels)
        pred.extend(model.predict(images, verbose=0).argmax(axis=1))
    return true, pred

def plot_confusion_matrix(true, pred, label_names):
    plt.figure(figsize=(10, 20))
    for normalize, i in zip(["true", "pred", "all", None], range(4)):
        plt.subplot(4, 2, 1+i)
        plt.title(f"Confusion Matrix (normalize={normalize})")
        plt.axis("off")
        ConfusionMatrixDisplay.from_predictions(
            y_true=true,
            y_pred=pred,
            display_labels=label_names,
            normalize=normalize,
            cmap="gray",
            im_kw= {"vmin": 0, "vmax": 1} if normalize is not None else None,
            ax=plt.gca()
        )

    plt.show()


def plot_scores(true, pred, label_names):
    f1 = f1_score(true, pred, average=None)
    precision = precision_score(true, pred, average=None)
    recall = recall_score(true, pred, average=None)

    plt.figure(figsize=(15, 5))

    for name, values, index in [("F1 Score", f1, 1), ("Precision", precision, 2), ("Recall", recall, 3)]:
        plt.subplot(1, 4, index + 1)
        plt.title(name)
        plt.bar(label_names, values, color=["red", "blue", "green", "orange", "purple"])
        plt.axis((-1, len(label_names), 0, 1))
        plt.xticks(rotation=-45)
