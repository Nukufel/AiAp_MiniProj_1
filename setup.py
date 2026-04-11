import kagglehub
from os import remove, mkdir
from os.path import exists
import shutil

DOWNLOAD_PATH = ".cache/extracted"
EXTRACTED_PATH = DOWNLOAD_PATH + "/Rice_Image_Dataset"
IMAGE_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

if exists(DOWNLOAD_PATH):
    shutil.rmtree(DOWNLOAD_PATH)

mkdir(DOWNLOAD_PATH)

kagglehub.dataset_download("muratkokludataset/rice-image-dataset", output_dir=DOWNLOAD_PATH)

shutil.rmtree(f"{DOWNLOAD_PATH}/.complete")

for label in IMAGE_LABELS:
    for i in range(2001, 15001) if label != "Karacadag" else range(151, 15001):
        path = f"{EXTRACTED_PATH}/{label}/{label} ({i}).jpg"
        # Basmati dataset has some files in lowercase
        path_alt = f"{EXTRACTED_PATH}/{label}/{label.lower()} ({i}).jpg"
        if exists(path):
            print(f"Deleting {label}-{i}")
            remove(path)

        if exists(path_alt):
            print(f"Deleting {label}-{i}")
            remove(path_alt)
