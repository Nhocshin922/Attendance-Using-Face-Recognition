import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ArcFace.mobile_model import mobileFaceNet, Arcloss
from tqdm import tqdm
import os
from torchsummary import summary


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 20
batch_size = 32  # Reduced batch size
learning_rate = 0.001
num_classes = 10  # Adjust based on your dataset

# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load your dataset
train_dataset = datasets.ImageFolder(root=r"C:\Users\ACER\Documents\InternProjects\Attendant Recognition\Face-recognition-using-transfer-learning-master\dataset\train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=r"C:\Users\ACER\Documents\InternProjects\Attendant Recognition\Face-recognition-using-transfer-learning-master\dataset\val", transform=val_transform)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load the pre-trained model
model = mobileFaceNet()
checkpoint = torch.load(r"C:\Users\ACER\Documents\InternProjects\Attendant Recognition\Face_Recognition_using_pytorch-master\Face_Recognition_using_pytorch-master\ArcFace\model\068.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['backbone_net_list'])
model = model.to(device)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few layers
for param in model.conv2.parameters():
    param.requires_grad = True
for param in model.linear7.parameters():
    param.requires_grad = True
for param in model.linear1.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = Arcloss(num_classes, s=30, m=0.5).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Training loop
best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), r"C:\Users\ACER\Documents\InternProjects\Attendant Recognition\Face_Recognition_using_pytorch-master\Face_Recognition_using_pytorch-master\ArcFace\model\069.pth")
        print("Saved best model!")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == patience:
        print('Early stopping!')
        break
    
    scheduler.step()

print("Training completed!")