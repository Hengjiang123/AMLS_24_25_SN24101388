import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class CNN_A(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_A, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # input channel=1, output channel=3, kernal=3x3
            nn.ReLU(),                                  # active function
            nn.MaxPool2d(kernel_size=2),  # 14 x 14     # polling layer with kernal 2x2, reduce output size 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),# input channel=16, output channel=32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 7 x 7       # reduce output size again 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),                  # all connect layer with input feature 32x7x7, output 64
            nn.ReLU(),
            nn.Dropout(0.3),                            # Dropout, Regularization
            nn.Linear(64, num_classes)                  # output classify results
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def dataloaders_A(train_images, train_labels, val_images, val_labels, batch_size=32):
    # make numpy data to Tensor type by DataLoader
    train_dataset = TensorDataset(
        torch.tensor(train_images, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_images, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def CNN_train_A(model, train_loader, val_loader, epochs=10, lr=1e-3, weight_decay=1e-4, device='cpu'):
    # train light CNN, and calculate four metrics for each epochs
    # use L2 regularization

    criterion = nn.CrossEntropyLoss()           # loss function: cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # store indexes for each epochs
    metrics_record = {
        'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [],
        'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': []
    }

    model.to(device)

    for epoch in range(epochs):
        model.train()                           # set model as training model
        all_preds = []
        all_labels = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()               # clear gradient
            outputs = model(X_batch)            # forward propagation   
            loss = criterion(outputs, y_batch)  # calculate loss
            loss.backward()                     # backward propagation
            optimizer.step()                    # update parameters

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels = y_batch.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
        
        # calculate metrics in train set
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(all_labels, all_preds)
        metrics_record['train_acc'].append(train_acc)
        metrics_record['train_prec'].append(train_prec)
        metrics_record['train_rec'].append(train_rec)
        metrics_record['train_f1'].append(train_f1)

        # validation dataset 
        model.eval()                            # set model as validate model
        val_preds = []
        val_labels = []
        with torch.no_grad():                   # no gradient used

            
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = y_batch.cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels)

        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(val_labels, val_preds)
        metrics_record['val_acc'].append(val_acc)
        metrics_record['val_prec'].append(val_prec)
        metrics_record['val_rec'].append(val_rec)
        metrics_record['val_f1'].append(val_f1)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, metrics_record
