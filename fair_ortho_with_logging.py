import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import logging
from datetime import datetime

class ChestXrayDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        """
        Args:
            parquet_file (str): Path to Parquet file with annotations
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform if transform else self.get_default_transforms()
        
        # Create mappings for race and gender
        self.race_mapping = {
            'WHITE': 0,
            'BLACK': 1,
            'OTHER': 2,
            'HISPANIC': 3,
            'ASIAN': 4,
            'NATIVE': 5
        }
        
        self.gender_mapping = {
            'M': 0,
            'F': 1
        }
    
    @staticmethod
    def get_default_transforms():
        """Default transformations for X-ray images"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet required size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path
        img_name = self.data.iloc[idx]['img_path']
        
        # Load and convert image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get labels
        disease_label = float(self.data.iloc[idx]['No Finding'])  # Convert to float for BCE loss
        race_label = self.race_mapping[self.data.iloc[idx]['race']]
        gender_label = self.gender_mapping[self.data.iloc[idx]['gender']]
        
        return {
            'image': image,
            'Path':img_name,
            'disease': torch.tensor(disease_label, dtype=torch.float),
            'race': torch.tensor(race_label, dtype=torch.long),
            'gender': torch.tensor(gender_label, dtype=torch.float)
        }

def create_data_loaders(parquet_file,batch_size=32, train_split=0.8):
    """
    Create train and validation dataloaders
    
    Args:
        parquet_file (str): Path to Parquet file
        batch_size (int): Batch size for dataloaders
        train_split (float): Proportion of data to use for training
    """
    # Create dataset
    dataset = ChestXrayDataset(parquet_file)
    
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_val_df_from_loader(val_loader):
    """Create a DataFrame from validation DataLoader"""
    all_data = []
    
    for batch_idx, batch in enumerate(val_loader):
        images = batch['image']
        path = batch['Path']
        disease_labels = batch['disease']
        race_labels = batch['race']
        gender_labels = batch['gender']
        
        # Convert tensors to lists/numpy arrays
        batch_size = images.size(0)
        for i in range(batch_size):
            sample_data = {
                'Path': path[i],
                'label': disease_labels[i].item(),
                'fitzpatrick': race_labels[i].item()+1,
            }
            all_data.append(sample_data)
    
    # Create DataFrame
    val_df = pd.DataFrame(all_data)
    return val_df

# Model Definition
class FairXRayClassifier(nn.Module):
    def __init__(self, num_races=6):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_dim = 2048

        # Remove final FC layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Disease branch
        self.disease_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        self.disease_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # Single neuron for binary classification (logits)
        )

        self.disease_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim)
        )

        # Race branch
        self.race_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        self.race_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_races),
        )

        self.race_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim)
        )
        
        # Freeze decoder weights
        for param in self.disease_decoder.parameters():
            param.requires_grad = False
        for param in self.race_decoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)

        # Disease predictions
        disease_features = self.disease_encoder(features)
        disease_pred = self.disease_head(disease_features)  # Outputs logits

        # Race predictions
        race_features = self.race_encoder(features)
        race_logits = self.race_head(race_features)

        # Decoded Features
        disease_decoded = self.disease_decoder(disease_features)
        race_decoded = self.race_decoder(race_features)

        return disease_pred, race_logits, disease_decoded, race_decoded, features


# Loss Function
def compute_loss(disease_pred, disease_labels, race_logits, race_labels, disease_decoded, race_decoded, feature):
    # print(f"Race Labels: {race_labels}")
    # print(f"Race Logits Shape: {race_logits.shape}")

    # Use binary_cross_entropy_with_logits for raw logits
    disease_loss = F.binary_cross_entropy_with_logits(disease_pred.view(-1), disease_labels.float().view(-1))

    # Convert race_labels to long and validate
    race_labels = race_labels.long()
    # print(f"Converted Race Labels: {race_labels}")
    # print(f"Min Label: {race_labels.min().item()}, Max Label: {race_labels.max().item()}")

    # Check for invalid labels
    num_classes = race_logits.shape[1]
    if torch.any(race_labels < 0) or torch.any(race_labels >= num_classes):
        invalid_labels = race_labels[(race_labels < 0) | (race_labels >= num_classes)]
        raise ValueError(f"Invalid race_labels detected: {invalid_labels}. Expected range: [0, {num_classes-1}]")

    # Compute race loss
    race_loss = F.cross_entropy(race_logits, race_labels)

    # Orthogonality Loss
    # orthogonal_loss = torch.norm(disease_decoded * race_decoded, p="fro")

    # Cosine Orthogonality Loss
    # Compute dot product along the feature dimension (dim=1)
    dot_product = (disease_decoded * race_decoded).sum(dim=1)  # Shape: [batch_size]
    norm_disease = torch.norm(disease_decoded, p=2, dim=1)     # Shape: [batch_size]
    norm_race = torch.norm(race_decoded, p=2, dim=1)          # Shape: [batch_size]
    
    # Cosine similarity (avoid division by zero with a small epsilon)
    epsilon = 1e-8
    cosine_similarity = dot_product / (norm_disease * norm_race + epsilon)  # Shape: [batch_size]
    orthogonal_loss = torch.abs(cosine_similarity).mean()  # Scalar: mean absolute cosine similarity
    
    constraint_loss = torch.abs(
        torch.norm(disease_decoded, dim=1, p=2) ** 2 +
        torch.norm(race_decoded, dim=1, p=2) ** 2 -
        torch.norm(feature, dim=1, p=2) ** 2
    ).mean()

    total_loss = disease_loss + race_loss + orthogonal_loss + 0.1 * constraint_loss
    return total_loss, disease_loss, race_loss, orthogonal_loss, constraint_loss



# Set up logging
def setup_logging(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return log_file

# Training Function with model saving and logging
def train(model, train_loader, val_loader, num_epochs=5, max_grad_norm=1.0, save_dir='checkpoints'):
    model.train()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define separate optimizers
    disease_params = list(model.disease_encoder.parameters()) + list(model.disease_head.parameters())
    race_params = list(model.race_encoder.parameters()) + list(model.race_head.parameters())
    shared_params = list(model.encoder.parameters())
    
    optimizer_disease = optim.Adam(disease_params, lr=1e-4, weight_decay=1e-6)
    optimizer_race = optim.Adam(race_params, lr=1e-4, weight_decay=1e-6)
    optimizer_shared = optim.Adam(shared_params, lr=1e-4, weight_decay=1e-6)
    
    best_val_acc = 0.0  # To track best model
    
    for epoch in range(num_epochs):
        total_loss = 0
        disease_correct, race_correct = 0, 0
        total = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            disease_labels = batch['disease'].to(device)
            race_labels = batch['race'].to(device)

            # Forward Pass 1: Disease Loss
            optimizer_disease.zero_grad()
            optimizer_shared.zero_grad()
            disease_pred, race_logits, disease_decoded, race_decoded, feature = model(images)
            _, disease_loss, _, _, _ = compute_loss(
                disease_pred, disease_labels, race_logits, race_labels, 
                disease_decoded, race_decoded, feature
            )
            disease_loss.backward()
            torch.nn.utils.clip_grad_norm_(disease_params + shared_params, max_grad_norm)
            optimizer_disease.step()
            optimizer_shared.step()

            # Forward Pass 2: Race Loss
            optimizer_race.zero_grad()
            optimizer_shared.zero_grad()
            disease_pred, race_logits, disease_decoded, race_decoded, feature = model(images)
            _, _, race_loss, _, _ = compute_loss(
                disease_pred, disease_labels, race_logits, race_labels, 
                disease_decoded, race_decoded, feature
            )
            race_loss.backward()
            torch.nn.utils.clip_grad_norm_(race_params + shared_params, max_grad_norm)
            optimizer_race.step()
            optimizer_shared.step()

            # Forward Pass 3: Constraint Loss
            optimizer_disease.zero_grad()
            optimizer_race.zero_grad()
            optimizer_shared.zero_grad()
            disease_pred, race_logits, disease_decoded, race_decoded, feature = model(images)
            _, _, _, orth_loss, constr_loss = compute_loss(
                disease_pred, disease_labels, race_logits, race_labels, 
                disease_decoded, race_decoded, feature
            )
            constraint = orth_loss + 0.1 * constr_loss
            constraint.backward()
            torch.nn.utils.clip_grad_norm_(disease_params + race_params + shared_params, max_grad_norm)
            optimizer_disease.step()
            optimizer_race.step()
            optimizer_shared.step()

            # Compute Accuracy
            disease_pred_binary = (disease_pred.squeeze() > 0).long()
            race_pred_labels = torch.argmax(race_logits, dim=1)

            disease_correct += (disease_pred_binary == disease_labels).sum().item()
            race_correct += (race_pred_labels == race_labels).sum().item()
            total += disease_labels.size(0)

            total_loss += (disease_loss + race_loss + constraint).item()

        disease_acc = 100 * disease_correct / total
        race_acc = 100 * race_correct / total
        
        # Validation
        val_disease_acc, val_race_acc = validate(model, val_loader)
        
        # Log training metrics
        logging.info(f"Epoch {epoch+1}/{num_epochs}:")
        logging.info(f"Train Loss: {total_loss:.4f}")
        logging.info(f"Train Disease Acc: {disease_acc:.2f}%")
        logging.info(f"Train Race Acc: {race_acc:.2f}%")
        logging.info(f"Val Disease Acc: {val_disease_acc:.2f}%")
        logging.info(f"Val Race Acc: {val_race_acc:.2f}%")
        logging.info(f"Disease Loss: {disease_loss.item():.4f}, Race Loss: {race_loss.item():.4f}, "
                    f"Ortho Loss: {orth_loss.item():.4f}, Constr Loss: {constr_loss.item():.4f}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_disease_state_dict': optimizer_disease.state_dict(),
            'optimizer_race_state_dict': optimizer_race.state_dict(),
            'optimizer_shared_state_dict': optimizer_shared.state_dict(),
            'train_loss': total_loss,
            'disease_acc': disease_acc,
            'race_acc': race_acc
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model based on average validation accuracy
        avg_val_acc = (val_disease_acc + val_race_acc) / 2
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_disease_state_dict': optimizer_disease.state_dict(),
                'optimizer_race_state_dict': optimizer_race.state_dict(),
                'optimizer_shared_state_dict': optimizer_shared.state_dict(),
                'train_loss': total_loss,
                'disease_acc': disease_acc,
                'race_acc': race_acc
            }, best_model_path)
            logging.info(f"Saved best model: {best_model_path} (Avg Val Acc: {avg_val_acc:.2f}%)")

# Validation Function with logging
def validate(model, val_loader):
    model.eval()
    disease_correct, race_correct = 0, 0
    total = 0
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            disease_labels = batch['disease'].to(device)
            race_labels = batch['race'].to(device)

            disease_pred, race_logits, disease_decoded, race_decoded, feature = model(images)
            
            # Compute validation loss
            total_loss, disease_loss, race_loss, orth_loss, constr_loss = compute_loss(
                disease_pred, disease_labels, race_logits, race_labels,
                disease_decoded, race_decoded, feature
            )
            total_val_loss += total_loss.item()

            disease_pred_binary = (disease_pred.squeeze() > 0).long()
            race_pred_labels = torch.argmax(race_logits, dim=1)

            disease_correct += (disease_pred_binary == disease_labels).sum().item()
            race_correct += (race_pred_labels == race_labels).sum().item()
            total += disease_labels.size(0)

    disease_acc = 100 * disease_correct / total
    race_acc = 100 * race_correct / total
    avg_val_loss = total_val_loss / len(val_loader)

    # Log validation metrics
    logging.info("Validation Metrics:")
    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Validation Disease Accuracy: {disease_acc:.2f}%")
    logging.info(f"Validation Race Accuracy: {race_acc:.2f}%")
    
    return disease_acc, race_acc

# Execution
if __name__ == "__main__":
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FairXRayClassifier().to(device)
    
    # Setup logging
    log_file = setup_logging()
    logging.info(f"Starting training with log file: {log_file}")
    logging.info(f"Using device: {device}")
    
    # Parameters
    csv_path = '/workspace/test/2k_sampled.parquet'
    batch_size = 32
    num_epochs = 10
    
    # Create dataloaders
    train_loader, val_loader = create_data_loaders(csv_path, batch_size)
    val_df = create_val_df_from_loader(val_loader)
    val_df.to_csv('val_dataset.csv', index=False)
    
    # Train the model
    train(model, train_loader, val_loader, num_epochs=num_epochs)