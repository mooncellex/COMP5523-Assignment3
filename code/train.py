import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader
from dataset import load_mnist, load_cifar10, load_fashion_mnist, imshow
from model import LeNet5

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#trainset, testset = load_cifar10()
trainset, testset = load_mnist()
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
testloader = DataLoader(testset, batch_size=16, shuffle=False)


model_sgd = LeNet5.to(device)
model_adam = LeNet5.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)

scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[50, 75], gamma=0.1)
scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[50, 75], gamma=0.1)

# test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


# loop over the dataset multiple times
num_epoch = 5

model_sgd.train()
model_adam.train()

for epoch in range(num_epoch):
    
    running_loss_sgd = 0.0
    running_loss_adam = 0.0
    
    for i, batch in enumerate(trainloader, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_sgd.zero_grad()  
        optimizer_adam.zero_grad()  

        # get prediction
        outputs_sgd = model_sgd(images)
        outputs_adam = model_adam(images)
        
        # compute loss
        loss_sgd = loss_fn(outputs_sgd, labels)
        loss_adam = loss_fn(outputs_adam, labels)
        
        # reduce loss
        loss_sgd.backward()
        loss_adam.backward()
        optimizer_sgd.step()
        optimizer_adam.step()

        # print statistics
        running_loss_sgd += loss_sgd.item()
        running_loss_adam += loss_adam.item()
        
        if i % 1000 == 999:  # print every 1000 mini-batches
            print('[epoch %2d, batch %5d] SGD loss: %.3f' %
                  (epoch + 1, i + 1, running_loss_sgd / 500))
            print('[epoch %2d, batch %5d] Adam loss: %.3f' %
                  (epoch + 1, i + 1, running_loss_adam / 500))
            running_loss_sgd = 0.0
            running_loss_adam = 0.0
            
            train_acc_sgd = accuracy(model_sgd, trainloader)
            test_acc_sgd = accuracy(model_sgd, testloader)
            print('SGD - Accuracy on the train set: %.2f %%' % (100 * train_acc_sgd))
            print('SGD - Accuracy on the test set: %.2f %%' % (100 * test_acc_sgd))
            print('SGD - Current learning rate: %.6f' % (scheduler_sgd.get_last_lr()[0]))
            
            train_acc_adam = accuracy(model_adam, trainloader)
            test_acc_adam = accuracy(model_adam, testloader)
            print('Adam - Accuracy on the train set: %.2f %%' % (100 * train_acc_adam))
            print('Adam - Accuracy on the test set: %.2f %%' % (100 * test_acc_adam))
            print('Adam - Current learning rate: %.6f' % (scheduler_adam.get_last_lr()[0]))

    scheduler_sgd.step()
    scheduler_adam.step()



model_file_sgd = 'model_sgd.pth'
model_file_adam = 'model_adam.pth'
torch.save(model_sgd.state_dict(), model_file_sgd)
torch.save(model_adam.state_dict(), model_file_adam)
print(f'SGD Model saved to {model_file_sgd}.')
print(f'Adam Model saved to {model_file_adam}.')

print('Finished Training')


# show some prediction result
dataiter = iter(testloader)
# images, labels = dataiter.next()
images, labels = next(dataiter)
images = images.to(device)

predictions_sgd = model_sgd(images).argmax(1).detach().cpu()
predictions_adam = model_adam(images).argmax(1).detach().cpu()

classes = trainset.classes
print('GroundTruth: ', ' '.join('%5s' % classes[i] for i in labels))
print('SGD - Prediction: ', ' '.join('%5s' % classes[i] for i in predictions_sgd))
print('Adam - Prediction: ', ' '.join('%5s' % classes[i] for i in predictions_adam))
imshow(torchvision.utils.make_grid(images.cpu()))


train_acc_sgd = accuracy(model_sgd, trainloader)
train_acc_adam = accuracy(model_adam, trainloader)
test_acc_sgd = accuracy(model_sgd, testloader)
test_acc_adam = accuracy(model_adam, testloader)

print('SGD - Accuracy on the train set: %f %%' % (100 * train_acc_sgd))
print('SGD - Accuracy on the test set: %f %%' % (100 * test_acc_adam))

print('Adam - Accuracy on the train set: %f %%' % (100 * train_acc_sgd))
print('Adam - Accuracy on the test set: %f %%' % (100 * test_acc_adam))
