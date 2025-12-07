import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .gnn import HiFiGAT
from .dataset import SmartContractDataset
import os
import random

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, prec, rec, f1

def train():
    # Configuration
    # Use absolute path or relative to this file to be safe
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "data", "solidity_bytecode")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "hifi_gat.pth")
    
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    
    print(f"Looking for data in: {DATASET_PATH}")
    
    # Check if data exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path {DATASET_PATH} does not exist. Please create it and add .bin/.evm files.")
        return

    # 1. Prepare Data
    full_dataset = SmartContractDataset(root=DATASET_PATH)
    # Filter out None entries (failed processing)
    dataset = [d for d in full_dataset if d is not None]
    
    if len(dataset) == 0:
        print("No valid data found.")
        return

    # Shuffle and Split
    random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - VAL_SPLIT))
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    
    print(f"Total samples: {len(dataset)}. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # num_node_features must match vocab_size in preprocess.py (150)
    model = HiFiGAT(num_node_features=150, hidden_channels=32, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Training Loop
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Validation
        if len(val_dataset) > 0:
            acc, prec, rec, f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")
            
            # Save best model
            if f1 >= best_f1:
                best_f1 = f1
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
            # Save anyway if no validation set
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
