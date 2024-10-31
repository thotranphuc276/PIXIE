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
args = parser.parse_args()

# Initialize device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pixie = PIXIE(config=pixie_cfg, device=device)
keypoint_loss_fn = KeypointLoss(confidence_threshold=0.3)

# Ensure that all parameters require gradients
params = []
for module in [pixie.Encoder, pixie.Regressor, pixie.Extractor, pixie.Moderator]:
    for net in module.values():
        for param in net.parameters():
            param.requires_grad = True
            params.append(param)

# Optimizer setup
optimizer = optim.Adam(params, lr=1e-4)

# DataLoader setup with random sampling and batch size
dataset = COCOWholeBodyDataset('coco_wholebody_train_v1.0.json')
if args.num_samples:
    # Generate random indices
    random_indices = random.sample(range(len(dataset)), args.num_samples)
    # Create a Subset with these random indices
    dataset = torch.utils.data.Subset(dataset, random_indices)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Training loop
for epoch in range(10):
    total_loss = 0

    for batch in tqdm(dataloader):
        # Move data to the appropriate device
        images = batch['image'].to(device)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass through PIXIE
        data = {'body': {'image': images, 'image_hd': images}}
        param_dict = pixie.encode(data)
        opdict = pixie.decode(param_dict['body'], param_type='body')

        # Compute loss and ensure it requires gradients
        loss_dict = keypoint_loss_fn.compute_loss(opdict, batch, keypoint_type='all')
        loss = loss_dict['total']
        loss = loss.requires_grad_(True)  # Ensure it tracks gradients

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/10], Loss: {total_loss:.4f}')

print("Training completed!")
