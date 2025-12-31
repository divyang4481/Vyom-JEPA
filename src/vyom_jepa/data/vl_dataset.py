import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class VLDataset(Dataset):
    """
    Multimodal Dataset for VL-JEPA.
    Returns:
        image: Tensor
        query: str
        target: str
    """

    def __init__(
        self, root_dir, phase="train", transform=None, dataset_type="synthetic"
    ):
        self.root = root_dir
        self.phase = phase
        self.transform = transform
        self.dataset_type = dataset_type
        self.samples = []

        if dataset_type == "vl_synthetic":
            # Generate fake data for testing pipeline
            for i in range(100):
                self.samples.append(
                    {"image": "dummy.jpg", "caption": f"A synthetic image number {i}"}
                )
        elif dataset_type == "flickr8k":
            # Expect standard Flickr8k structure:
            # root_dir/Images/*.jpg
            # root_dir/captions.txt (image,caption)

            captions_file = os.path.join(self.root, "captions.txt")
            if not os.path.exists(captions_file):
                raise FileNotFoundError(
                    f"Flickr8k captions not found at {captions_file}. Please download the dataset."
                )

            with open(captions_file, "r", encoding="utf-8") as f:
                # Skip header if exists (image,caption)
                lines = f.readlines()
                # Basic CSV parse
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        img_name = parts[0]
                        caption = ",".join(parts[1:])  # Re-join if comma in caption

                        # Check header
                        if img_name.lower() == "image":
                            continue

                        self.samples.append({"image": img_name, "caption": caption})
            print(f"Loaded {len(self.samples)} samples from Flickr8k.")
        else:
            raise ValueError(f"Unknown dataset {dataset_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Image
        if self.dataset_type == "vl_synthetic":
            # Create random noise image
            img = Image.fromarray(
                torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
            )
        else:
            img_path = os.path.join(self.root, "Images", item["image"])
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # 2. Query & Target
        # Pretraining: Query = "Describe image", Target = Caption
        query = "Describe the image."
        target = item["caption"]

        return {"image": img, "query": query, "target": target}
