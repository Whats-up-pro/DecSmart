# =============================================================================
# SOLIDIFI BENCHMARK - OPTIMIZED TRAINING FOR OVERFLOW-UNDERFLOW & RE-ENTRANCY
# =============================================================================
# Pipeline tối ưu cho 2 loại vulnerability:
# 1. Overflow-Underflow (Arithmetic)
# 2. Re-entrancy
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import random
import numpy as np
import urllib.request
import zipfile
import io
import json
from collections import Counter
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CELL 1: CONFIGURATION & SETUP
# =============================================================================
print("=" * 80)
print("🔧 CONFIGURATION - OPTIMIZED FOR OVERFLOW-UNDERFLOW & RE-ENTRANCY")
print("=" * 80)

# Seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Directories
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data", "solidifi_optimized")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Target vulnerabilities (focused training)
TARGET_VULNS = ['Overflow-Underflow', 'Re-entrancy']
VULN_LABELS = {
    'Overflow-Underflow': 0,  # Arithmetic vulnerability
    'Re-entrancy': 1,         # Reentrancy vulnerability
}
NUM_CLASSES = 2

# Optimized Hyperparameters
HYPERPARAMS = {
    # Model Architecture
    'num_node_features': 64,      # Reduced for focused patterns
    'hidden_channels': 128,       # Sufficient capacity
    'num_heads': 4,               # Optimized attention
    'num_layers': 4,              # Deeper network
    'dropout': 0.35,              # Strong regularization
    
    # Training
    'learning_rate': 0.0008,      # Optimized LR
    'weight_decay': 0.01,         # L2 regularization
    'batch_size': 16,             # Smaller for better gradients
    'epochs': 200,                # More epochs
    'patience': 30,               # Early stopping
    
    # Loss
    'focal_gamma': 2.5,           # Increased for hard samples
    'label_smoothing': 0.05,      # Smoothing factor
    
    # Class weights (tuned for balance)
    'class_weights': {
        'Overflow-Underflow': 1.5,
        'Re-entrancy': 2.0,       # Higher weight for Re-entrancy
    },
    
    # Thresholds (optimized)
    'thresholds': {
        'Overflow-Underflow': 0.40,  # Lower threshold
        'Re-entrancy': 0.35,         # Lower threshold for higher recall
    }
}

print("\n📋 Hyperparameters:")
for k, v in HYPERPARAMS.items():
    print(f"   {k}: {v}")

# =============================================================================
# CELL 2: DATA ACQUISITION - SOLIDIFI-BENCHMARK
# =============================================================================
print("\n" + "=" * 80)
print("📥 DOWNLOADING SOLIDIFI-BENCHMARK DATA")
print("=" * 80)

SOLIDIFI_URL = "https://github.com/DependableSystemsLab/SolidiFI-benchmark/archive/refs/heads/master.zip"
SOLIDIFI_DIR = os.path.join(DATA_DIR, "SolidiFI-benchmark")

def download_solidifi():
    """Download and extract SolidiFI-benchmark"""
    if os.path.exists(SOLIDIFI_DIR):
        print("   ✅ SolidiFI-benchmark already exists")
        return True
    
    print("   📥 Downloading SolidiFI-benchmark...")
    try:
        response = urllib.request.urlopen(SOLIDIFI_URL, timeout=180)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall(DATA_DIR)
        
        # Rename extracted folder
        extracted = os.path.join(DATA_DIR, "SolidiFI-benchmark-master")
        if os.path.exists(extracted):
            import shutil
            if os.path.exists(SOLIDIFI_DIR):
                shutil.rmtree(SOLIDIFI_DIR)
            os.rename(extracted, SOLIDIFI_DIR)
        
        print("   ✅ Downloaded successfully")
        return True
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False

download_solidifi()

# =============================================================================
# CELL 3: ENHANCED FEATURE EXTRACTION
# =============================================================================
print("\n" + "=" * 80)
print("⚙️ FEATURE EXTRACTION - ENHANCED FOR TARGET VULNERABILITIES")
print("=" * 80)

class EnhancedFeatureExtractor:
    """
    Enhanced feature extraction specifically for:
    - Overflow-Underflow: Arithmetic operations without SafeMath
    - Re-entrancy: External calls before state changes
    """
    
    def __init__(self, num_features=64):
        self.num_features = num_features
        
        # Pattern definitions with weights
        self.patterns = {
            # ============ OVERFLOW-UNDERFLOW PATTERNS ============
            'arithmetic_add': (r'\+\s*(?!=)', 3),         # + but not +=
            'arithmetic_sub': (r'-\s*(?!=)', 3),          # - but not -=
            'arithmetic_mul': (r'\*\s*(?!=)', 3),         # * but not *=
            'arithmetic_div': (r'/\s*(?!=)', 2),          # / but not /=
            'arithmetic_exp': (r'\*\*', 4),               # Exponentiation
            'unchecked_block': (r'unchecked\s*\{', 5),    # Explicit unchecked
            'no_safemath': (r'using\s+SafeMath', -5),     # SafeMath usage (negative = good)
            'uint_overflow': (r'uint\d*\s+\w+\s*[+\-*]', 4),  # Direct uint arithmetic
            'increment': (r'\+\+|\+=', 2),                # Increment operators
            'decrement': (r'--|-=', 2),                   # Decrement operators
            'loop_counter': (r'for\s*\([^)]*\+\+[^)]*\)', 3),  # Loop with counter
            'balance_calc': (r'balance\s*[+\-*\/]', 4),   # Balance calculations
            'amount_calc': (r'amount\s*[+\-*\/]', 4),     # Amount calculations
            
            # ============ RE-ENTRANCY PATTERNS ============
            'external_call': (r'\.call\s*\{?', 5),        # External call
            'call_value': (r'\.call\{value:', 6),         # call with value
            'call_value_old': (r'\.call\.value\s*\(', 6), # Old call.value syntax
            'send_ether': (r'\.send\s*\(', 4),            # send()
            'transfer_ether': (r'\.transfer\s*\(', 3),    # transfer()
            'delegatecall': (r'\.delegatecall\s*\(', 5),  # delegatecall
            'state_after_call': (r'\.call[^;]*;\s*\n\s*\w+\s*=', 7),  # State change after call
            'balance_before_call': (r'balance\s*=.*\.call', 5),  # Pattern for reentrancy
            'withdraw_pattern': (r'function\s+withdraw.*\.call', 6),  # Withdraw + call
            'msg_value': (r'msg\.value', 3),              # msg.value usage
            'payable': (r'\bpayable\b', 2),               # payable function
            'receive_function': (r'receive\s*\(\s*\)', 2),  # receive function
            'fallback': (r'fallback\s*\(\s*\)', 3),       # fallback function
            
            # ============ COMMON PATTERNS ============
            'require': (r'require\s*\(', -1),             # Input validation
            'assert': (r'assert\s*\(', -1),               # Assertion
            'revert': (r'revert\s*\(', -1),               # Revert
            'modifier': (r'modifier\s+\w+', -1),          # Modifier definition
            'onlyowner': (r'onlyOwner', -1),              # Access control
            'nonreentrant': (r'nonReentrant|ReentrancyGuard', -5),  # Reentrancy guard
            'check_effects': (r'require.*;\s*\w+\s*=.*;\s*\.call', -3),  # Check-effects pattern
            'lock_mutex': (r'locked\s*=\s*true', -3),     # Mutex lock
            'reentrancy_check': (r'if\s*\(\s*!?locked\s*\)', -3),  # Lock check
            
            # ============ STRUCTURAL PATTERNS ============
            'function_count': (r'function\s+\w+', 1),     # Number of functions
            'public_func': (r'\bpublic\b', 1),            # Public visibility
            'external_func': (r'\bexternal\b', 2),        # External visibility
            'internal_func': (r'\binternal\b', 0),        # Internal visibility
            'mapping': (r'mapping\s*\(', 1),              # State variable (mapping)
            'storage': (r'\bstorage\b', 1),               # Storage reference
            'memory': (r'\bmemory\b', 0),                 # Memory reference
            'interface': (r'\binterface\b', 1),           # Interface usage
            'import': (r'import\s+', 1),                  # Import statement
        }
    
    def extract_features(self, source_code, vuln_type=None):
        """Extract feature vector from Solidity source code"""
        features = []
        source_lower = source_code.lower()
        
        # Pattern-based features
        for pattern_name, (pattern, weight) in self.patterns.items():
            matches = len(re.findall(pattern, source_code, re.IGNORECASE | re.MULTILINE))
            # Apply weight and normalize
            weighted_count = matches * abs(weight) / 10.0
            if weight < 0:
                weighted_count = -weighted_count
            features.append(min(weighted_count, 1.0))
        
        # Additional statistical features
        lines = source_code.split('\n')
        features.append(min(len(lines) / 500.0, 1.0))  # Normalized line count
        features.append(min(len(source_code) / 10000.0, 1.0))  # Normalized char count
        
        # Function complexity
        functions = re.findall(r'function\s+\w+[^{]*\{[^}]*\}', source_code, re.DOTALL)
        avg_func_len = np.mean([len(f) for f in functions]) if functions else 0
        features.append(min(avg_func_len / 500.0, 1.0))
        
        # Nesting depth estimation
        max_depth = max(line.count('{') - line.count('}') for line in lines) if lines else 0
        features.append(min(max_depth / 5.0, 1.0))
        
        # Vulnerability-specific boost
        if vuln_type == 'Overflow-Underflow':
            # Boost arithmetic-related features
            has_safemath = bool(re.search(r'using\s+SafeMath', source_code, re.IGNORECASE))
            has_unchecked = bool(re.search(r'unchecked\s*\{', source_code))
            has_arithmetic = bool(re.search(r'[+\-*/]', source_code))
            features.append(1.0 if (has_arithmetic and not has_safemath) else 0.0)
            features.append(1.0 if has_unchecked else 0.0)
        elif vuln_type == 'Re-entrancy':
            # Boost reentrancy-related features
            has_call = bool(re.search(r'\.call', source_code))
            has_guard = bool(re.search(r'nonReentrant|ReentrancyGuard|locked', source_code, re.IGNORECASE))
            state_after_call = bool(re.search(r'\.call[^;]*;[^}]*\w+\s*=', source_code, re.DOTALL))
            features.append(1.0 if (has_call and not has_guard) else 0.0)
            features.append(1.0 if state_after_call else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Pad or truncate to num_features
        while len(features) < self.num_features:
            features.append(0.0)
        features = features[:self.num_features]
        
        return np.array(features, dtype=np.float32)
    
    def code_to_graph(self, source_code, vuln_type=None):
        """Convert Solidity code to PyG Data object"""
        # Extract base features
        base_features = self.extract_features(source_code, vuln_type)
        
        # Create multiple nodes (function-level granularity)
        functions = re.findall(r'function\s+\w+[^{]*\{[^}]*\}', source_code, re.DOTALL)
        num_nodes = max(1, min(len(functions), 10))
        
        node_features = []
        for i in range(num_nodes):
            if i < len(functions):
                func_features = self.extract_features(functions[i], vuln_type)
                # Blend function and contract features
                blended = 0.7 * func_features + 0.3 * base_features
            else:
                blended = base_features + np.random.randn(self.num_features) * 0.05
            node_features.append(np.clip(blended, 0, 1))
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Create edges (fully connected + self-loops)
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)

# Initialize extractor
feature_extractor = EnhancedFeatureExtractor(num_features=HYPERPARAMS['num_node_features'])
print(f"   ✅ Feature extractor initialized with {HYPERPARAMS['num_node_features']} features")

# =============================================================================
# CELL 4: LOAD AND PROCESS DATA
# =============================================================================
print("\n" + "=" * 80)
print("📂 LOADING SOLIDIFI-BENCHMARK DATA")
print("=" * 80)

def load_solidifi_contracts(target_vulns=TARGET_VULNS):
    """Load contracts from SolidiFI-benchmark for target vulnerabilities"""
    contracts = []
    buggy_dir = os.path.join(SOLIDIFI_DIR, "buggy_contracts")
    
    if not os.path.exists(buggy_dir):
        print(f"   ❌ Directory not found: {buggy_dir}")
        return contracts
    
    for vuln_folder in os.listdir(buggy_dir):
        vuln_path = os.path.join(buggy_dir, vuln_folder)
        
        if not os.path.isdir(vuln_path):
            continue
        
        # Check if this is a target vulnerability
        if vuln_folder not in target_vulns:
            continue
        
        for sol_file in os.listdir(vuln_path):
            if not sol_file.endswith('.sol'):
                continue
            
            filepath = os.path.join(vuln_path, sol_file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                if len(code) < 100:  # Skip very short files
                    continue
                
                contracts.append({
                    'filename': sol_file,
                    'code': code,
                    'vuln_type': vuln_folder,
                    'label': VULN_LABELS[vuln_folder]
                })
            except Exception as e:
                continue
    
    return contracts

# Load contracts
all_contracts = load_solidifi_contracts()
print(f"\n   ✅ Loaded {len(all_contracts)} contracts")

# Distribution
vuln_counts = Counter([c['vuln_type'] for c in all_contracts])
print("\n   📊 Distribution:")
for vuln, count in vuln_counts.items():
    print(f"      {vuln}: {count} contracts")

# Convert to graphs
print("\n   🔄 Converting to graphs...")
all_graphs = []
graph_labels = []

for contract in tqdm(all_contracts, desc="   Processing"):
    try:
        graph = feature_extractor.code_to_graph(contract['code'], contract['vuln_type'])
        graph.y = torch.tensor([contract['label']], dtype=torch.long)
        all_graphs.append(graph)
        graph_labels.append(contract['label'])
    except Exception as e:
        continue

print(f"   ✅ Converted {len(all_graphs)} graphs")

# =============================================================================
# CELL 5: DATA SPLITTING (80/20 with stratification)
# =============================================================================
print("\n" + "=" * 80)
print("📊 DATA SPLITTING - 80% TRAIN / 20% TEST")
print("=" * 80)

# Stratified split
labels = np.array(graph_labels)
indices = np.arange(len(all_graphs))

train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2, 
    stratify=labels, 
    random_state=SEED
)

# Further split train into train/val (90/10)
train_labels = labels[train_idx]
train_idx_final, val_idx = train_test_split(
    train_idx,
    test_size=0.1,
    stratify=train_labels,
    random_state=SEED
)

train_graphs = [all_graphs[i] for i in train_idx_final]
val_graphs = [all_graphs[i] for i in val_idx]
test_graphs = [all_graphs[i] for i in test_idx]

print(f"   Train: {len(train_graphs)} graphs")
print(f"   Validation: {len(val_graphs)} graphs")
print(f"   Test: {len(test_graphs)} graphs")

# Distribution check
for name, graphs in [('Train', train_graphs), ('Val', val_graphs), ('Test', test_graphs)]:
    counts = Counter([g.y.item() for g in graphs])
    dist = {list(VULN_LABELS.keys())[k]: v for k, v in counts.items()}
    print(f"   {name} distribution: {dist}")

# =============================================================================
# CELL 6: DATA AUGMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("🔄 DATA AUGMENTATION")
print("=" * 80)

def augment_graph(graph, noise_level=0.1, dropout_prob=0.1):
    """Augment graph with noise and feature dropout"""
    new_x = graph.x.clone()
    
    # Add Gaussian noise
    noise = torch.randn_like(new_x) * noise_level
    new_x = new_x + noise
    
    # Feature dropout
    mask = torch.rand_like(new_x) > dropout_prob
    new_x = new_x * mask.float()
    
    # Clamp values
    new_x = torch.clamp(new_x, 0, 1)
    
    return Data(
        x=new_x, 
        edge_index=graph.edge_index.clone(), 
        y=graph.y.clone()
    )

# Augment training data (2x augmentation)
augmented_train = []
for g in train_graphs:
    augmented_train.append(g)
    augmented_train.append(augment_graph(g, 0.08, 0.1))
    augmented_train.append(augment_graph(g, 0.12, 0.15))

print(f"   Original train: {len(train_graphs)}")
print(f"   Augmented train: {len(augmented_train)} (3x)")

# =============================================================================
# CELL 7: MODEL DEFINITION - OPTIMIZED GAT
# =============================================================================
print("\n" + "=" * 80)
print("🧠 MODEL DEFINITION - OPTIMIZED GAT")
print("=" * 80)

class OptimizedGAT(nn.Module):
    """
    Optimized GAT for Overflow-Underflow and Re-entrancy detection
    """
    def __init__(self, num_features, hidden_channels, num_classes, 
                 heads=4, dropout=0.35, num_layers=4):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Input projection with LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # GAT layers with skip connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels * heads
            self.gat_layers.append(
                GATConv(in_ch, hidden_channels, heads=heads, 
                       dropout=dropout, concat=True, add_self_loops=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_channels * heads))
            self.skip_projs.append(nn.Linear(in_ch, hidden_channels * heads))
        
        # Final projection before pooling
        self.pre_pool = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )
        
        # Multi-scale pooling weights (learnable)
        self.pool_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch):
        # Input projection
        x = self.input_proj(x)
        
        # GAT layers with residual connections
        for i in range(self.num_layers):
            identity = self.skip_projs[i](x)
            x = self.gat_layers[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.gelu(x)
            x = x + identity  # Residual connection
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Pre-pooling projection
        x = self.pre_pool(x)
        
        # Multi-scale pooling with learnable weights
        pool_w = F.softmax(self.pool_weights, dim=0)
        x_mean = global_mean_pool(x, batch) * pool_w[0]
        x_max = global_max_pool(x, batch) * pool_w[1]
        x_add = global_add_pool(x, batch) * pool_w[2]
        
        # Concatenate pooled features
        graph_emb = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Classification
        return self.classifier(graph_emb)

# Create model
model = OptimizedGAT(
    num_features=HYPERPARAMS['num_node_features'],
    hidden_channels=HYPERPARAMS['hidden_channels'],
    num_classes=NUM_CLASSES,
    heads=HYPERPARAMS['num_heads'],
    dropout=HYPERPARAMS['dropout'],
    num_layers=HYPERPARAMS['num_layers']
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {num_params:,}")
print(f"   Architecture: {HYPERPARAMS['num_layers']} GAT layers, {HYPERPARAMS['num_heads']} heads")

# =============================================================================
# CELL 8: LOSS FUNCTION - FOCAL LOSS WITH LABEL SMOOTHING
# =============================================================================
print("\n" + "=" * 80)
print("📉 LOSS FUNCTION - FOCAL LOSS WITH LABEL SMOOTHING")
print("=" * 80)

class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with Label Smoothing for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.5, smoothing=0.05, num_classes=2):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.alpha = alpha  # Class weights
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Cross entropy with focal weight
        ce_loss = -targets_smooth * log_probs * focal_weight
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce_loss = ce_loss * alpha_t.unsqueeze(1)
        
        return ce_loss.sum(dim=1).mean()

# Class weights based on configuration
class_weights = torch.tensor([
    HYPERPARAMS['class_weights']['Overflow-Underflow'],
    HYPERPARAMS['class_weights']['Re-entrancy']
], dtype=torch.float).to(device)

criterion = FocalLossWithSmoothing(
    alpha=class_weights,
    gamma=HYPERPARAMS['focal_gamma'],
    smoothing=HYPERPARAMS['label_smoothing'],
    num_classes=NUM_CLASSES
)

print(f"   Focal gamma: {HYPERPARAMS['focal_gamma']}")
print(f"   Label smoothing: {HYPERPARAMS['label_smoothing']}")
print(f"   Class weights: {class_weights.cpu().numpy()}")

# =============================================================================
# CELL 9: TRAINING LOOP
# =============================================================================
print("\n" + "=" * 80)
print("🚀 TRAINING LOOP")
print("=" * 80)

# Create data loaders
train_loader = DataLoader(augmented_train, batch_size=HYPERPARAMS['batch_size'], shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=HYPERPARAMS['batch_size'], shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=HYPERPARAMS['batch_size'], shuffle=False)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=HYPERPARAMS['learning_rate'],
    weight_decay=HYPERPARAMS['weight_decay']
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# Training state
best_val_f1 = 0.0
best_state = None
patience_counter = 0

# Training history
history = {
    'train_loss': [], 'val_f1': [], 
    'val_f1_ofu': [], 'val_f1_re': []
}

def evaluate(model, loader, thresholds=None):
    """Evaluate model on a data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Apply custom thresholds if provided
    if thresholds:
        custom_preds = []
        for i in range(len(all_probs)):
            # Get probability for each class
            prob_ofu = all_probs[i, 0]
            prob_re = all_probs[i, 1]
            
            # Apply thresholds
            if prob_ofu >= thresholds['Overflow-Underflow']:
                custom_preds.append(0)
            elif prob_re >= thresholds['Re-entrancy']:
                custom_preds.append(1)
            else:
                custom_preds.append(all_preds[i])
        all_preds = np.array(custom_preds)
    
    # Calculate metrics per class
    results = {}
    for label, name in enumerate(['Overflow-Underflow', 'Re-entrancy']):
        mask = all_labels == label
        if mask.sum() > 0:
            class_preds = (all_preds == label).astype(int)
            class_labels = (all_labels == label).astype(int)
            
            results[name] = {
                'precision': precision_score(class_labels, class_preds, zero_division=0),
                'recall': recall_score(class_labels, class_preds, zero_division=0),
                'f1': f1_score(class_labels, class_preds, zero_division=0),
                'support': int(mask.sum())
            }
    
    # Macro F1
    macro_f1 = np.mean([r['f1'] for r in results.values()])
    
    return results, macro_f1, all_preds, all_labels

# Training loop
print(f"\n   Starting training for {HYPERPARAMS['epochs']} epochs...")
print(f"   Patience: {HYPERPARAMS['patience']}")
print("-" * 60)

for epoch in range(HYPERPARAMS['epochs']):
    # Training phase
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_loss)
    
    # Validation phase
    if (epoch + 1) % 5 == 0 or epoch == 0:
        results, val_f1, _, _ = evaluate(model, val_loader, HYPERPARAMS['thresholds'])
        
        history['val_f1'].append(val_f1)
        history['val_f1_ofu'].append(results.get('Overflow-Underflow', {}).get('f1', 0))
        history['val_f1_re'].append(results.get('Re-entrancy', {}).get('f1', 0))
        
        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            
            print(f"   Epoch {epoch+1:>3} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | "
                  f"OFU: {results.get('Overflow-Underflow', {}).get('f1', 0):.3f} | "
                  f"RE: {results.get('Re-entrancy', {}).get('f1', 0):.3f} ✓")
        else:
            patience_counter += 1
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1:>3} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | "
                      f"OFU: {results.get('Overflow-Underflow', {}).get('f1', 0):.3f} | "
                      f"RE: {results.get('Re-entrancy', {}).get('f1', 0):.3f}")
        
        # Early stopping
        if patience_counter >= HYPERPARAMS['patience']:
            print(f"\n   ⚠️ Early stopping at epoch {epoch+1}")
            break

# Load best model
if best_state:
    model.load_state_dict(best_state)

print(f"\n   ✅ Training complete! Best Val F1: {best_val_f1:.4f}")

# =============================================================================
# CELL 10: THRESHOLD OPTIMIZATION
# =============================================================================
print("\n" + "=" * 80)
print("🔧 THRESHOLD OPTIMIZATION")
print("=" * 80)

def optimize_thresholds(model, val_loader):
    """Find optimal thresholds for each class"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    optimized_thresholds = {}
    
    for class_idx, class_name in enumerate(['Overflow-Underflow', 'Re-entrancy']):
        best_th = 0.5
        best_f1 = 0
        
        # Search for best threshold
        for th in np.arange(0.25, 0.75, 0.02):
            preds = (all_probs[:, class_idx] >= th).astype(int)
            labels = (all_labels == class_idx).astype(int)
            
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        
        optimized_thresholds[class_name] = best_th
        print(f"   {class_name}: Optimal threshold = {best_th:.2f} (F1: {best_f1:.4f})")
    
    return optimized_thresholds

optimized_thresholds = optimize_thresholds(model, val_loader)

# =============================================================================
# CELL 11: FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("📊 FINAL EVALUATION ON TEST SET")
print("=" * 80)

# Evaluate with optimized thresholds
results, test_f1, test_preds, test_labels = evaluate(model, test_loader, optimized_thresholds)

print("\n" + "=" * 65)
print(f"   {'Vulnerability':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
print("-" * 65)

for name, metrics in results.items():
    print(f"   {name:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
          f"{metrics['f1']:>10.4f} {metrics['support']:>10}")

print("-" * 65)
print(f"   {'MACRO AVERAGE':<20} {'':<10} {'':<10} {test_f1:>10.4f}")
print("=" * 65)

# =============================================================================
# CELL 12: CONFUSION MATRIX & VISUALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("📈 CONFUSION MATRIX")
print("=" * 80)

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Overflow-Underflow', 'Re-entrancy'],
            yticklabels=['Overflow-Underflow', 'Re-entrancy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Overflow-Underflow & Re-entrancy Detection')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'confusion_matrix_ofu_re.png'), dpi=150)
plt.show()

# Training History
if len(history['val_f1']) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # F1 curves
    epochs = list(range(0, len(history['val_f1']) * 5, 5))
    axes[1].plot(epochs, history['val_f1'], label='Macro F1', linewidth=2)
    axes[1].plot(epochs, history['val_f1_ofu'], label='Overflow-Underflow F1', linestyle='--')
    axes[1].plot(epochs, history['val_f1_re'], label='Re-entrancy F1', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Scores')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_history_ofu_re.png'), dpi=150)
    plt.show()

# =============================================================================
# CELL 13: SAVE MODEL
# =============================================================================
print("\n" + "=" * 80)
print("💾 SAVING MODEL")
print("=" * 80)

save_path = os.path.join(MODEL_SAVE_DIR, 'optimized_gat_ofu_re.pt')

torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparams': HYPERPARAMS,
    'vuln_labels': VULN_LABELS,
    'optimized_thresholds': optimized_thresholds,
    'results': results,
    'test_f1': test_f1,
    'feature_extractor_config': {
        'num_features': HYPERPARAMS['num_node_features']
    }
}, save_path)

print(f"   ✅ Model saved to: {save_path}")

# =============================================================================
# CELL 14: SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("📋 TRAINING SUMMARY")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────┐
│                    OPTIMIZED GAT TRAINING RESULTS              │
├────────────────────────────────────────────────────────────────┤
│ Target Vulnerabilities:                                        │
│   • Overflow-Underflow (Arithmetic)                            │
│   • Re-entrancy                                                │
├────────────────────────────────────────────────────────────────┤
│ Dataset: SolidiFI-benchmark                                    │
│   • Train: {len(train_graphs):>4} graphs (augmented to {len(augmented_train)})               │
│   • Validation: {len(val_graphs):>4} graphs                                   │
│   • Test: {len(test_graphs):>4} graphs                                       │
├────────────────────────────────────────────────────────────────┤
│ Model Architecture:                                            │
│   • GAT Layers: {HYPERPARAMS['num_layers']}                                           │
│   • Attention Heads: {HYPERPARAMS['num_heads']}                                       │
│   • Hidden Channels: {HYPERPARAMS['hidden_channels']}                                     │
│   • Dropout: {HYPERPARAMS['dropout']}                                          │
├────────────────────────────────────────────────────────────────┤
│ Optimized Thresholds:                                          │
│   • Overflow-Underflow: {optimized_thresholds.get('Overflow-Underflow', 0.5):.2f}                             │
│   • Re-entrancy: {optimized_thresholds.get('Re-entrancy', 0.5):.2f}                                   │
├────────────────────────────────────────────────────────────────┤
│ Final Results (Test Set):                                      │
│   • Overflow-Underflow F1: {results.get('Overflow-Underflow', {}).get('f1', 0):.4f}                        │
│   • Re-entrancy F1: {results.get('Re-entrancy', {}).get('f1', 0):.4f}                              │
│   • Macro F1: {test_f1:.4f}                                       │
└────────────────────────────────────────────────────────────────┘
""")

print("✅ Training pipeline complete!")
