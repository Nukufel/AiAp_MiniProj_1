from keras.utils import image_dataset_from_directory
from tensorflow.data.experimental import cardinality

RAW_DATASET_CACHE = ".cache/extracted/Rice_Image_Dataset"
SEED = 96

def get_data(image_size = (32, 32)):
    (train_data, test_data) = image_dataset_from_directory(
        RAW_DATASET_CACHE,
        image_size=image_size,
        validation_split=0.3,
        subset="both",
        seed=SEED
    )
    train_size = cardinality(train_data).numpy()
    test_size = cardinality(test_data).numpy()

    train_size = train_data.shuffle(10000, seed=SEED).take(int(train_size * 0.3))
    test_size = test_data.shuffle(10000, seed=SEED).take(int(test_size * 0.3))

    return train_data, test_data