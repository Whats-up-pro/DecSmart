import torch
import os
import argparse
from torch_geometric.loader import DataLoader
from .gnn import HiFiGAT
from .dataset import SmartContractDataset
from .preprocess import CFGBuilder

def test_model(data_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # num_node_features must match vocab_size in preprocess.py (150)
    model = HiFiGAT(num_node_features=150, hidden_channels=32, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Prepare Data
    if os.path.isdir(data_path):
        dataset = SmartContractDataset(root=data_path)
        dataset = [d for d in dataset if d is not None]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        print(f"Testing on {len(dataset)} samples from {data_path}")
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.exp(out)
                pred = out.argmax(dim=1).item()
                label = batch.y.item()
                
                status = "CORRECT" if pred == label else "WRONG"
                print(f"Sample {i}: Pred={pred} (Prob: {probs[0][1]:.4f}), True={label} -> {status}")
                
                if pred == label:
                    correct += 1
                total += 1
        
        if total > 0:
            print(f"Accuracy: {correct/total:.4f} ({correct}/{total})")
            
    elif os.path.isfile(data_path):
        # Single file test
        builder = CFGBuilder()
        with open(data_path, 'r') as f:
            bytecode = f.read().strip()
            
        try:
            instructions = builder.disassemble(bytecode)
            cfg = builder.build_cfg(instructions)
            data = builder.graph_to_data(cfg)
            
            # Add batch dimension
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            data = data.to(device)
            
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.batch)
                probs = torch.exp(out)
                pred = out.argmax(dim=1).item()
                
            print(f"File: {data_path}")
            print(f"Prediction: {'VULNERABLE' if pred == 1 else 'SAFE'}")
            print(f"Confidence (Vuln): {probs[0][1]:.4f}")
            
        except Exception as e:
            print(f"Error processing file: {e}")

if __name__ == "__main__":
    # Default paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_DATA = os.path.join(BASE_DIR, "data", "solidity_bytecode")
    DEFAULT_MODEL = os.path.join(BASE_DIR, "saved_models", "hifi_gat.pth")
    
    parser = argparse.ArgumentParser(description='Test HiFi-GAT Model')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA, help='Path to data folder or single file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Path to saved model')
    
    args = parser.parse_args()
    
    test_model(args.data, args.model)
