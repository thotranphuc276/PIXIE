import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class COCOWholeBodyDataset:
    def __init__(self, annotation_path, cache_dir='image_cache'):
        # Load annotations from JSON file
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        self.image_info = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert image to tensor
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get annotation and image info
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_info = self.image_info[image_id]

        # Download and transform the image
        image = self.load_or_download_image(image_info['coco_url'], image_id)

        # Extract keypoints and bounding boxes with default fallback to zeros
        data = {
            'image': image,
            'body_keypoints': self._extract_keypoints(annotation, 'keypoints', 17),
            'foot_keypoints': self._extract_keypoints(annotation, 'foot_kpts', 6),
            'face_keypoints': self._extract_keypoints(annotation, 'face_kpts', 68),
            'lefthand_keypoints': self._extract_keypoints(annotation, 'lefthand_kpts', 21),
            'righthand_keypoints': self._extract_keypoints(annotation, 'righthand_kpts', 21),
            'face_box': torch.tensor(annotation.get('face_box', [0, 0, 0, 0])),
            'lefthand_box': torch.tensor(annotation.get('lefthand_box', [0, 0, 0, 0])),
            'righthand_box': torch.tensor(annotation.get('righthand_box', [0, 0, 0, 0])),
        }

        return data

    def load_or_download_image(self, url, image_id):
        """Load image from cache or download and save if not available."""
        cache_path = os.path.join(self.cache_dir, f"{image_id}.pt")

        if os.path.exists(cache_path):
            # Load the image tensor from cache and convert to PIL Image
            image_tensor = torch.load(cache_path)
            image = transforms.ToPILImage()(image_tensor)
            # Apply resize transformation
            image = transforms.Resize((224, 224))(image)
            # Convert back to tensor
            return transforms.ToTensor()(image)

        # Download the image if not in cache
        image_tensor = self.download_image(url)
        # Save the image tensor to cache
        torch.save(image_tensor, cache_path)

        return image_tensor

    def _extract_keypoints(self, annotation, key, expected_len):
        """Extract keypoints or return default if missing."""
        keypoints = annotation.get(key, [0] * (expected_len * 3))
        return torch.tensor(keypoints, dtype=torch.float32).reshape(-1, 3)

    def download_image(self, url):
        """Downloads an image and converts it to a tensor."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return torch.zeros(3, 224, 224)
