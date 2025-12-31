import os
import requests
import zipfile
import sys


def main():
    print("--- Flickr8k Setup Helper ---")
    data_dir = "data/flickr8k"
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "Images")

    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 100:
        print("Flickr8k seems to be present.")
        return

    print(
        "Note: Automatic download of Flickr8k (1GB) via direct link is unreliable due to licensing."
    )
    print("Please follow these steps:")
    print(
        f"1. Download 'captions.txt' and 'Images' from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k"
    )
    print(f"2. Place 'captions.txt' in: {os.path.abspath(data_dir)}")
    print(f"3. Place images in: {os.path.abspath(images_dir)}")
    print(
        "\nOnce done, you can restart training with 'dataset_type: flickr8k' in config.yaml"
    )


if __name__ == "__main__":
    main()
