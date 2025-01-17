import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, ResNet34_Weights
from B.Evaluation_B import calculate_metrics_B

class ResNet34_B(nn.Module):
    # use resnet43 in torchvision and modify fianl full-connected layer to 8 classes
    # use preprocessed train dataset Tenser with 3 channels
    def __init__(self, num_classes=8, weights=ResNet34_Weights.IMAGENET1K_V1):
        super(ResNet34_B, self).__init__()
        self.model = resnet34(weights=weights)
        # Modify the final fc layer: change its output to 8
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def ResNet_train_B(model, train_loader, val_loader, epochs=10, lr=1e-3, weight_decay=1e-4, device='cpu'):
    # train RenNet34, calculate and record four metrics in each epochs:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    metrics_record = {
        'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [],
        'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_f1': [],
        'train_loss': [], 'val_loss': []  
    }

    model.to(device)

    # training part
    for epoch in range(epochs):
        model.train()
        all_preds = []
        all_labels = []
        total_train_loss = 0.0
        num_train_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_train_loss = total_train_loss / num_train_batches
        metrics_record['train_loss'].append(avg_train_loss)
        
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics_B(all_labels, all_preds)
        metrics_record['train_acc'].append(train_acc)
        metrics_record['train_prec'].append(train_prec)
        metrics_record['train_rec'].append(train_rec)
        metrics_record['train_f1'].append(train_f1)

        # validation part
        model.eval()
        val_preds = []
        val_true = []
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                num_val_batches += 1

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = y_batch.cpu().numpy()

                val_preds.extend(preds)
                val_true.extend(labels)

        avg_val_loss = total_val_loss / num_val_batches
        metrics_record['val_loss'].append(avg_val_loss)

        val_acc, val_prec, val_rec, val_f1 = calculate_metrics_B(val_true, val_preds)
        metrics_record['val_acc'].append(val_acc)
        metrics_record['val_prec'].append(val_prec)
        metrics_record['val_rec'].append(val_rec)
        metrics_record['val_f1'].append(val_f1)

        print(f"Epoch [{epoch+1}/{epochs}] -> "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, metrics_record
