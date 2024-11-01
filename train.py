import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from pixielib.pixie import PIXIE
from pixielib.utils.config import cfg as pixie_cfg
from loader import COCOWholeBodyDataset
from keypoint_loss import KeypointLoss
import random

torch.cuda.empty_cache()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train PIXIE model on COCOWholeBody dataset.")
parser.add_argument('--num_samples', type=int, default=None, help="Number of random samples to train on")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train for")
args = parser.parse_args()

# Initialize device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pixie = PIXIE(config=pixie_cfg, device=device)
keypoint_loss_fn = KeypointLoss(confidence_threshold=0.3)

# Ensure all parameters require gradients
params = []
for module in [pixie.Encoder, pixie.Regressor, pixie.Extractor, pixie.Moderator]:
    for net in module.values():
        for param in net.parameters():
            param.requires_grad = True
            params.append(param)

# Optimizer setup
optimizer = optim.Adam(params, lr=1e-4)

# DataLoader setup for cached and non-cached images
dataset = COCOWholeBodyDataset('coco_wholebody_train_v1.0.json')
cached_dataset = torch.utils.data.Subset(dataset, range(len(dataset.cached_annotations)))
non_cached_dataset = torch.utils.data.Subset(dataset, range(len(dataset.cached_annotations), len(dataset)))

# Use DataLoader to prioritize cached images first
cached_dataloader = torch.utils.data.DataLoader(cached_dataset, batch_size=args.batch_size, shuffle=True)
non_cached_dataloader = torch.utils.data.DataLoader(non_cached_dataset, batch_size=args.batch_size, shuffle=True)

num_epochs = args.num_epochs

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    # Train on cached images first
    for batch in tqdm(cached_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}] - Cached Images"):
        images = batch['image'].to(device)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        data = {'body': {'image': images, 'image_hd': images}}
        param_dict = pixie.encode(data)
        opdict = pixie.decode(param_dict['body'], param_type='body')

        loss_dict = keypoint_loss_fn.compute_loss(opdict, batch, keypoint_type='all')
        loss = loss_dict['total'].requires_grad_(True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

print("Training completed!")
