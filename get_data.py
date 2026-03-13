from huggingface_hub import snapshot_download
import zipfile

snapshot_download(repo_id="nateraw/rice-image-dataset", repo_type="dataset", local_dir="raw_dataset")
with zipfile.ZipFile("raw_dataset/rice-image-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")