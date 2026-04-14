import os
import kagglehub
import shutil

DOWNLOAD_PATH = '.cache/extracted'
EXTRACTED_PATH = DOWNLOAD_PATH + '/seg_train/seg_train'
IMAGE_LABELS = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

if os.path.exists(DOWNLOAD_PATH):
    shutil.rmtree(DOWNLOAD_PATH)

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

kagglehub.dataset_download(
    'puneet6060/intel-image-classification', output_dir=DOWNLOAD_PATH
)

shutil.rmtree(f'{DOWNLOAD_PATH}/.complete', ignore_errors=True)


def extract_number(filename):
    # "66.jpg" -> 66
    return int(os.path.splitext(filename)[0])


for label in IMAGE_LABELS:
    folder = os.path.join(EXTRACTED_PATH, label)

    # Get all jpg files
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

    # Sort numerically (IMPORTANT)
    files.sort(key=extract_number)

    keep = 150 if label == 'street' else 2000

    for f in files[keep:]:
        path = os.path.join(folder, f)
        if os.path.exists(path):
            print(f'Deleting {label}/{f}')
            os.remove(path)
