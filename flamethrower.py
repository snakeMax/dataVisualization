import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = "..\\..\\mnist\\trainingSet\\trainingSet"


dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))


model = SimpleNN().to('cpu')

logits = model(images)


# max_batches = 200 ## Number of batches of images to train (above 100 runs slow)
# num_epochs = 3 ## Number of times to loop through the dataset

def train(max_batches, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = min(len(dataloader), max_batches)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (images, labels) in enumerate(dataloader, 1):
            if i > max_batches:
                break
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress = int(40 * i / total_batches)
            bar = '[' + '#' * progress + '-' * (40 - progress) + ']'
            print(f"\r{bar} {i}/{total_batches} batches", end='', flush=True)
        
        avg_loss = running_loss / total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"model_epoch_{num_epochs}_batches_{max_batches}.pth")
    print("Model saved.")

    print("Training complete.")

def test(max_test_batches, num_epochs, batches): ## Limit how many batches to test on
    model.load_state_dict(torch.load(f"model_epoch_{num_epochs}_batches_{batches}.pth"))
    print("Model loaded for testing.")
    correct = 0
    total = 0  
    model.eval()  # Set model to evaluation mode


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, 1):
            if batch_idx > max_test_batches:
                break
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                is_correct = predicted[i].item() == labels[i].item()
                print(
                    f"[Test Batch {batch_idx} | Image {i+1}] "
                    f"Label: {labels[i].item()} | Prediction: {predicted[i].item()} | "
                    f"{'Correct' if is_correct else 'Wrong'}"
                )
                total += 1
                if is_correct:
                    correct += 1

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Tested {total} images. Test Accuracy: {accuracy:.2f}%")

Max_batches = 100 ## Number of batches of images to train (above 100 runs slow)
Num_epochs = 3 ## Number of times to loop through the dataset

if (not os.path.exists(f"model_epoch_{Num_epochs}_batches_{Max_batches}.pth")):
    print("No model found. Training...")
    train(Max_batches, Num_epochs)
elif (os.path.exists(f"model_epoch_{Num_epochs}_batches_{Max_batches}.pth")):
    print("Model found. Testing...")
    test(10, Num_epochs, Max_batches)
else:
    print("Error at running training/testing")

