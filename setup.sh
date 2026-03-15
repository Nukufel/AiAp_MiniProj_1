
mkdir -p ".cache"

cd ".cache" || exit

if [ -e "original.zip" ]; then
    echo "Dataset already downloaded, skipping download"
    else
    echo "Downloading rice image dataset..."
    curl "https://www.muratkoklu.com/datasets/Rice_Image_Dataset.zip" > "original.zip"
fi

echo "You are ready to go!"