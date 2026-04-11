import numpy as np
from keras.utils import image_dataset_from_directory
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
import tensorflow as tf
import random

RAW_DATASET_CACHE = ".cache/extracted/seg_train/seg_train"

SEED = 96
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TEST_SPLIT = 0.3
TRAIN_TO_VALIDATION_SPLIT = 0.8

TRAIN_VALIDATION_SPLIT = 1 - TEST_SPLIT

TRAIN_SPLIT = TRAIN_VALIDATION_SPLIT * TRAIN_TO_VALIDATION_SPLIT
VALIDATION_SPLIT = TRAIN_VALIDATION_SPLIT * (1 - TRAIN_TO_VALIDATION_SPLIT)


def split_dataset(dataset, train_split, validation_split, test_split):
    dataset_size = dataset.cardinality().numpy()

    train_size = int(train_split * dataset_size)
    validation_size = int(validation_split * dataset_size)
    test_size = int(test_split * dataset_size)

    train_samples = dataset.take(train_size)
    validation_test_samples = dataset.skip(train_size)

    validation_samples = validation_test_samples.take(validation_size)
    test_samples = validation_test_samples.skip(validation_size).take(test_size)

    return train_samples, validation_samples, test_samples

def get_data(image_size):
    data = image_dataset_from_directory(
        RAW_DATASET_CACHE,
        image_size=image_size,
        # For splitting we want to have it as accurate as possible
        batch_size=None,
        verbose=False,
        shuffle=True,
        seed=SEED,
    )

    label_names = data.class_names

    train_samples, validation_samples, test_samples = split_dataset(
        data,
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TEST_SPLIT,
    )

    print("Number of training images: ", len(train_samples))
    print("Number of validation images: ", len(validation_samples))
    print("Number of testing images: ", len(test_samples))
    print("Class names: ", label_names)

    return data, train_samples, validation_samples, test_samples, label_names

def plot_samples(train_images, label_names):
    plt.figure(figsize=(8, 4))

    for i, (images, labels) in enumerate(train_images.take(8)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images.numpy().astype("uint8"))
        plt.title(label_names[labels])
        plt.axis("off")
    plt.show()

def plot_number_per_class(title, images, label_names):
    number_per_class = [0, 0, 0, 0, 0, 0]

    for _, label in images.as_numpy_iterator():
        number_per_class[label] += 1

    plt.figure(figsize=(5, 5))
    plt.pie(number_per_class, labels=label_names, autopct=lambda x: round(x * images.cardinality().numpy() / 100))
    plt.title(title)
    # show number in pie chart

    plt.show()

def plot_accuracy_and_loss(accuracy, validation_accuracy, loss, validation_loss):
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label="training accuracy")
    plt.plot(validation_accuracy, label="validation accuracy")
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="training loss")
    plt.plot(validation_loss, label="validation loss")
    plt.yscale("log")
    plt.ylabel("Accuracy")
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
    plt.figure(figsize=(17, 30))
    for normalize, i in zip(["true", "pred", "all", None], range(4)):
        plt.subplot(4, 2, 1+i)
        ax=plt.gca()
        ConfusionMatrixDisplay.from_predictions(
            y_true=true,
            y_pred=pred,
            display_labels=label_names,
            normalize=normalize,
            cmap="gray",
            im_kw= {"vmin": 0, "vmax": 1} if normalize in ["true", "pred"] else None,
            ax=ax
        )

        ax.tick_params(axis="x", labelrotation=45)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(f"Confusion Matrix (normalize={normalize})")

    plt.show()



def plot_scores(true, pred, label_names: list[str]):
    f1 = f1_score(true, pred, average=None)
    precision = precision_score(true, pred, average=None)
    recall = recall_score(true, pred, average=None)

    plt.figure(figsize=(15, 5))

    for name, values, index in [("F1 Score", f1, 1), ("Precision", precision, 2), ("Recall", recall, 3)]:
        plt.subplot(1, 4, index + 1)
        plt.title(name)
        bars = plt.bar(x=label_names, height=values, color=["red", "blue", "green", "orange", "purple"])
        plt.axis((-1, len(label_names), 0, 1))
        plt.xticks(rotation=-45)
        plt.bar_label(container=bars, labels=[round(v, 2) for v in values], padding=-15)


get_data(image_size = (64, 64))