import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class NNTrainer():
    '''
    Class for training models
    '''
    def __init__(self, train_loader, valid_loader, test_loader, num_classes, batch_size, num_epochs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = 0.0005
        self.start_learning_rate = 0.0005
        self.num_epochs = num_epochs
        
        self.device = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
    
    def train(self, model: nn.Module, path: str):
        '''
        trains model and saves to path
        '''
        ac_l = []
        los = []
        
        self.learning_rate = self.start_learning_rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay = 0.005, momentum = 0.9)

        total_step = len(self.train_loader)

        for epoch in range(self.num_epochs):
            print("training")
            for i, (images, labels) in enumerate(self.train_loader):
                
                if (i % 1000 == 0):
                    print(i)
                
                images = images.to(torch.float32)
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                # print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                
                los.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, self.num_epochs, i + 1, total_step, loss.item()))

            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in self.valid_loader:
                    images = images.to(torch.float32)
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    del images, labels, outputs
                print(total, correct)
                accuracy = correct / total
                ac_l.append(accuracy)
                print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * accuracy))
        
        if epoch % 5 == 4:
            self.learning_rate /= 5
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay = 0.005, momentum = 0.9)
        
        torch.save(model.state_dict(), path)
        
        np.save("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/los.npy", np.array(los))
        np.save("/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/data/ac.npy", np.array(ac_l))