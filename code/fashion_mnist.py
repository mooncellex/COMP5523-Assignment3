import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

from torch.utils.data import DataLoader
from dataset import load_fashion_mnist
from torchvision.models import resnet18

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

trainset, testset = load_fashion_mnist()
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


## your code here
# TODO: load ResNet18 from PyTorch Hub, and train it to achieve 90+% classification accuracy on Fashion-MNIST.

# Preprocessing: Resize images to fit ResNet18 input size and apply data augmentation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    # Randomly flip images horizontally
    transforms.RandomHorizontalFlip(),
    # Randomly rotate images by up to 10 degrees
    transforms.RandomRotation(10), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Convert tensor to PIL image
to_pil = ToPILImage()

# Load ResNet18 from PyTorch Hub
model = resnet18(pretrained=False, num_classes=10)  # Set num_classes to 10 for Fashion-MNIST
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images_pil = [to_pil(img) for img in images]
        images = torch.stack([preprocess(img) for img in images_pil]).to(device)
        #images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

# Training loop
for epoch in range(10):  # You can adjust the number of epochs
    model.train()
    for images, labels in trainloader:
        # Convert tensor to PIL image
        images_pil = [to_pil(img) for img in images]

        # Preprocess the PIL images
        images = torch.stack([preprocess(img) for img in images_pil]).to(device)

        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item()}")
    
# Evaluate model accuracy
model.eval()

torch.save(model.state_dict(), "fashion_mnist.pth")
print("Model saved to fashion_mnist.pth.")

train_acc = accuracy(model, trainloader)
test_acc = accuracy(model, testloader)

print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy on the test set: %f %%' % (100 * test_acc))

