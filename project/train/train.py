import torch
import torch.nn as nn

class NNTrainer():
    def __init__(self, train_loader, valid_loader, test_loader, num_classes, batch_size, num_epochs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = 0.005
        self.start_learning_rate = 0.005
        self.num_epochs = num_epochs
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def train(self, model: nn.Module):
        self.learning_rate = self.start_learning_rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay = 0.005, momentum = 0.9)

        total_step = len(self.train_loader)

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in self.valid_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    del images, labels, outputs
                print(total, correct)
                accuracy = correct / total
                print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * accuracy))
        
        if epoch % 5 == 4:
            self.learning_rate /= 5
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay = 0.005, momentum = 0.9)