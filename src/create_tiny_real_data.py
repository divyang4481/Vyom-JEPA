import os
import requests
import time
from PIL import Image
import numpy as np

def download_image(url, filename):
    print(f"Downloading {url} to {os.path.abspath(filename)}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Success: {filename}")
            return True
    except Exception as e:
        print(f"Failed: {e}")
    return False

def main():
    root = "data/flickr8k"
    img_dir = os.path.join(root, "Images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Real reliable URLs
    samples = [
        # (Filename, Caption, URL)
        ("real_dog.jpg", "A dog sitting on the grass", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"),
        ("real_cat.jpg", "A cat lying on the bed", "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"),
        ("real_food.jpg", "A delicious pizza", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_it-na_pizza-margherita_sep2005_sml.jpg/320px-Eq_it-na_pizza-margherita_sep2005_sml.jpg"),
        ("real_city.jpg", "A city street at night", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall_%28cropped%29.jpg/320px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall_%28cropped%29.jpg"),
        ("real_plane.jpg", "An airplane flying in the sky", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Widebody_meeting_%284695955146%29.jpg/320px-Widebody_meeting_%284695955146%29.jpg")
    ]
    
    captions = []
    
    for filename, caption, url in samples:
        path = os.path.join(img_dir, filename)
        if download_image(url, path):
            captions.append(f"{filename},{caption}")
            time.sleep(1.0)
            
    # Write captions.txt
    with open(os.path.join(root, "captions.txt"), "w") as f:
        f.write("image,caption\n")
        for line in captions:
            f.write(line + "\n")
            
    print("Created Placeholder Real Dataset at data/flickr8k (images are random noise)")

if __name__ == "__main__":
    main()
