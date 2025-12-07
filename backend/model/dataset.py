import os
import torch
import json
from torch_geometric.data import Dataset
from .preprocess import CFGBuilder

class SmartContractDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.builder = CFGBuilder()
        super(SmartContractDataset, self).__init__(root, transform, pre_transform)
        self.file_list = []
        self.labels = {}
        
        # Load labels if available
        label_path = os.path.join(root, 'labels.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                self.labels = json.load(f)
        
        # Assume root contains subfolders 'vulnerable' and 'clean' or similar
        # For this template, we'll just look for .bin or .evm files in root
        if os.path.exists(root):
            for file in os.listdir(root):
                if file.endswith('.bin') or file.endswith('.evm'):
                    self.file_list.append(os.path.join(root, file))

    @property
    def raw_file_names(self):
        return self.file_list

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.file_list))]

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # In a real scenario, we would load the processed .pt file
        # Here we process on-the-fly for simplicity if processed doesn't exist
        # or just return what we have.
        
        # For the template, let's assume we read the raw file and process it
        file_path = self.file_list[idx]
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r') as f:
            bytecode = f.read().strip()
            
        try:
            instructions = self.builder.disassemble(bytecode)
            cfg = self.builder.build_cfg(instructions)
            data = self.builder.graph_to_data(cfg)
            
            # Determine label
            if filename in self.labels:
                label = self.labels[filename]
            elif 'vuln' in filename.lower():
                label = 1
            else:
                # Default to 0 (Safe) if unknown, but this is risky for training
                label = 0
                
            data.y = torch.tensor([label], dtype=torch.long)
            
            return data
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a dummy empty graph to avoid crashing
            return None
