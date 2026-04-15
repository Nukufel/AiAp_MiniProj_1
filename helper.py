from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from tensorflow.python.data.ops.dataset_ops import DatasetV2
import numpy as np
from statistics import mean, stdev

RAW_DATASET_CACHE = '.cache/extracted/seg_train/seg_train'

SHUFFLE_AMOUNT = 100000
SEED = 96

TEST_PERCENTAGE = 0.3
TRAIN_VALIDATION_PERCENTAGE = 0.7
TRAIN_TO_VALIDATION_RATIO = 0.8
TRAINING_PERCENTAGE = TRAIN_VALIDATION_PERCENTAGE * TRAIN_TO_VALIDATION_RATIO
VALIDATION_PERCENTAGE = TRAIN_VALIDATION_PERCENTAGE * (
    1 - TRAIN_TO_VALIDATION_RATIO
)


def split_dataset(
    dataset: DatasetV2,
) -> tuple[DatasetV2, DatasetV2, DatasetV2]:
    dataset_size = dataset.cardinality().numpy()

    train_size = int(TRAINING_PERCENTAGE * dataset_size)
    validation_size = int(VALIDATION_PERCENTAGE * dataset_size)
    test_size = int(TEST_PERCENTAGE * dataset_size)

    train_samples = dataset.take(train_size)
    validation_test_samples = dataset.skip(train_size)

    validation_samples = validation_test_samples.take(validation_size)
    test_samples = validation_test_samples.skip(validation_size).take(
        test_size
    )

    return train_samples, validation_samples, test_samples


def get_data(
    image_size,
) -> tuple[DatasetV2, DatasetV2, DatasetV2, DatasetV2, list[str]]:
    data = image_dataset_from_directory(
        RAW_DATASET_CACHE,
        image_size=image_size,
        # For splitting we want to have it as accurate as possible
        batch_size=None,
        shuffle=False,
        verbose=False,
    )

    label_names = data.class_names
    shuffled_data = data.shuffle(
        SHUFFLE_AMOUNT, seed=SEED, reshuffle_each_iteration=False
    )

    train_samples, validation_samples, test_samples = split_dataset(
        shuffled_data
    )

    print('Number of training images: ', len(train_samples))
    print('Number of validation images: ', len(validation_samples))
    print('Number of testing images: ', len(test_samples))
    print('Class names: ', label_names)

    return (
        shuffled_data,
        train_samples,
        validation_samples,
        test_samples,
        label_names,
    )


def plot_samples(train_images, label_names):
    plt.figure(figsize=(8, 4))

    for i, (images, labels) in enumerate(train_images.take(8)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images.numpy().astype('uint8'))
        plt.title(label_names[labels])
        plt.axis('off')
    plt.show()


def plot_numbers_per_classes(image_dict, label_names):
    plt.figure(figsize=(20, 20))
    for index, (title, images) in enumerate(image_dict.items()):
        plt.subplot(1, len(image_dict), 1 + index)
        plot_number_per_class(title, images, label_names)
    plt.show()


def plot_number_per_class(title, images, label_names):
    number_per_class = [0] * len(label_names)

    for _, label in images.as_numpy_iterator():
        number_per_class[label] += 1

    plt.pie(
        number_per_class,
        labels=label_names,
        autopct=lambda x: round(x * images.cardinality().numpy() / 100),
    )
    plt.title(title)


def print_accuracy_and_loss(model, images):
    result = model.evaluate(images, return_dict=True, verbose=0)
    print(f'Validation accuracy: {result["accuracy"]}')
    print(f'Validation Loss: {result["loss"]}')


def plot_accuracy_and_loss(
    accuracy, validation_accuracy, loss, validation_loss, log: bool = True
):
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label='training accuracy')
    plt.plot(validation_accuracy, label='validation accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    if log:
        plt.yscale('log')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


def calculate_predictions(model, images):
    true = np.array([label for _, label in images.unbatch()])
    pred_raw = model.predict(images, verbose=0)
    pred = pred_raw.argmax(axis=1)

    return true, pred, pred_raw


def plot_confusion_matrix(true, pred, label_names):
    plt.figure(figsize=(17, 30))
    for normalize, i in zip(['true', 'pred', 'all', None], range(4)):
        plt.subplot(4, 2, 1 + i)
        ax = plt.gca()
        ConfusionMatrixDisplay.from_predictions(
            y_true=true,
            y_pred=pred,
            display_labels=label_names,
            normalize=normalize,
            cmap='gray',
            im_kw={'vmin': 0, 'vmax': 1}
            if normalize in ['true', 'pred']
            else None,
            ax=ax,
        )

        ax.tick_params(axis='x', labelrotation=45)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'{i+1}. Confusion Matrix (normalize={normalize})')

    plt.show()


def plot_scores(true, pred, label_names: list[str]):
    f1 = f1_score(true, pred, average=None)
    precision = precision_score(true, pred, average=None)
    recall = recall_score(true, pred, average=None)

    plt.figure(figsize=(15, 5))

    for name, values, index in [
        ('F1 Score', f1, 1),
        ('Precision', precision, 2),
        ('Recall', recall, 3),
    ]:
        plt.subplot(1, 4, index + 1)
        plt.title(name)
        bars = plt.bar(
            x=label_names,
            height=values,
            color=['red', 'blue', 'green', 'orange', 'purple'],
        )
        plt.axis((-1, len(label_names), 0, 1))
        plt.xticks(rotation=-45)
        plt.bar_label(
            container=bars, labels=[round(v, 2) for v in values], padding=-15
        )


def dataset_to_sklearn(dataset: DatasetV2):
    images = []
    labels = []

    for image, label in dataset.as_numpy_iterator():
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels).flatten()


def execute_cv(create_model, dataset, folds, epochs):
    results = {
        'Accuracy': [],
        'Loss': [],
    }

    x, y = dataset_to_sklearn(dataset)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        print(f'Fold {fold + 1} / {folds}')
        model = create_model()

        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        history = model.fit(
            x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=epochs
        )

        evalu_result = model.evaluate(
            x=x_val, y=y_val, return_dict=True, verbose=0
        )

        results['Accuracy'].append(evalu_result['accuracy'])
        results['Loss'].append(evalu_result['loss'])

    return results


def plot_cv_results(results):
    plt.figure(figsize=(20, 8))

    for index, (name, values) in enumerate(results.items()):
        plt.subplot(1, len(results), index + 1)
        plt.title(name)
        plt.ylim([0, 1])

        bar = plt.bar(x=range(len(values) + 1)[1:], height=values)
        plt.bar_label(bar, labels=[round(v, 2) for v in values], padding=-15)
        plt.figlegend(['Mean'])

        plt.plot([0.5, 5.5], [mean(values)] * 2, label='Mean', color='red')

    plt.show()
