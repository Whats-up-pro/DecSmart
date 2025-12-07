import torch
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from typing import List, Dict, Tuple

# Try to import pyevmasm, handle if missing
try:
    from pyevmasm import disassemble_all, instruction_tables
except ImportError:
    print("Warning: pyevmasm not installed. Please install it via 'pip install pyevmasm'")
    disassemble_all = None

class CFGBuilder:
    """
    Constructs a High-Fidelity Control Flow Graph from EVM Bytecode.
    """
    def __init__(self, embedding_model_path=None):
        self.opcode_embeddings = {}
        # Define a fixed vocabulary for opcodes to use with Embedding layer
        self.opcode_vocab = {
            'STOP': 0, 'ADD': 1, 'MUL': 2, 'SUB': 3, 'DIV': 4, 'SDIV': 5, 'MOD': 6, 'SMOD': 7,
            'ADDMOD': 8, 'MULMOD': 9, 'EXP': 10, 'SIGNEXTEND': 11, 'LT': 12, 'GT': 13, 'SLT': 14,
            'SGT': 15, 'EQ': 16, 'ISZERO': 17, 'AND': 18, 'OR': 19, 'XOR': 20, 'NOT': 21,
            'BYTE': 22, 'SHL': 23, 'SHR': 24, 'SAR': 25, 'SHA3': 26, 'ADDRESS': 27, 'BALANCE': 28,
            'ORIGIN': 29, 'CALLER': 30, 'CALLVALUE': 31, 'CALLDATALOAD': 32, 'CALLDATASIZE': 33,
            'CALLDATACOPY': 34, 'CODESIZE': 35, 'CODECOPY': 36, 'GASPRICE': 37, 'EXTCODESIZE': 38,
            'EXTCODECOPY': 39, 'RETURNDATASIZE': 40, 'RETURNDATACOPY': 41, 'EXTCODEHASH': 42,
            'BLOCKHASH': 43, 'COINBASE': 44, 'TIMESTAMP': 45, 'NUMBER': 46, 'DIFFICULTY': 47,
            'GASLIMIT': 48, 'CHAINID': 49, 'SELFBALANCE': 50, 'BASEFEE': 51, 'POP': 52,
            'MLOAD': 53, 'MSTORE': 54, 'MSTORE8': 55, 'SLOAD': 56, 'SSTORE': 57, 'JUMP': 58,
            'JUMPI': 59, 'PC': 60, 'MSIZE': 61, 'GAS': 62, 'JUMPDEST': 63, 'PUSH1': 64,
            'PUSH2': 65, 'PUSH3': 66, 'PUSH4': 67, 'PUSH32': 95, 'DUP1': 96, 'DUP16': 111,
            'SWAP1': 112, 'SWAP16': 127, 'LOG0': 128, 'LOG4': 132, 'CREATE': 133, 'CALL': 134,
            'CALLCODE': 135, 'RETURN': 136, 'DELEGATECALL': 137, 'CREATE2': 138, 'STATICCALL': 139,
            'REVERT': 140, 'INVALID': 141, 'SELFDESTRUCT': 142
        }
        self.vocab_size = 150 # Reserve space for unknown opcodes
        
        if embedding_model_path:
            self.load_embeddings(embedding_model_path)

    def load_embeddings(self, path):
        # Load pre-trained Word2Vec model
        model = Word2Vec.load(path)
        self.opcode_embeddings = model.wv

    def disassemble(self, bytecode_hex: str):
        if not disassemble_all:
            raise ImportError("pyevmasm is required for disassembly")
        
        # Convert hex string to bytes
        if bytecode_hex.startswith("0x"):
            bytecode_hex = bytecode_hex[2:]
        bytecode = bytes.fromhex(bytecode_hex)
        
        return list(disassemble_all(bytecode))

    def build_cfg(self, instructions) -> nx.DiGraph:
        """
        Builds a CFG from a list of instructions.
        Nodes are basic blocks.
        """
        cfg = nx.DiGraph()
        
        # 1. Identify Basic Blocks
        # A basic block ends with a JUMP, JUMPI, STOP, RETURN, REVERT, INVALID, or SELFDESTRUCT
        # Or if the next instruction is a JUMPDEST
        
        blocks = []
        current_block = []
        pc_to_block_index = {} # Map PC of first instruction to block index
        
        terminators = {'JUMP', 'JUMPI', 'STOP', 'RETURN', 'REVERT', 'INVALID', 'SELFDESTRUCT'}
        
        for instr in instructions:
            # If current instruction is JUMPDEST and we have a non-empty current block, 
            # finish the current block first (unless it's the start)
            if instr.name == 'JUMPDEST' and current_block:
                blocks.append(current_block)
                current_block = []
            
            current_block.append(instr)
            
            if instr.name in terminators:
                blocks.append(current_block)
                current_block = []
        
        if current_block:
            blocks.append(current_block)

        # 2. Create Nodes and Map PCs
        for idx, block in enumerate(blocks):
            if not block: continue
            start_pc = block[0].pc
            pc_to_block_index[start_pc] = idx
            
            # Feature extraction for the block (simplified for now)
            # In HiFi-CFG, we aggregate features of instructions in the block
            cfg.add_node(idx, instructions=block, start_pc=start_pc)

        # 3. Create Edges
        for idx, block in enumerate(blocks):
            if not block: continue
            last_instr = block[-1]
            
            # Fallthrough edge (if not a hard stop/jump)
            if last_instr.name not in {'STOP', 'RETURN', 'REVERT', 'INVALID', 'SELFDESTRUCT', 'JUMP'}:
                if idx + 1 < len(blocks):
                    cfg.add_edge(idx, idx + 1, type='fallthrough')

            # Jump edges
            if last_instr.name in {'JUMP', 'JUMPI'}:
                # Resolve jump target
                # This is the hard part in static analysis (resolving dynamic jumps).
                # For this implementation, we look for PUSH instructions immediately preceding the JUMP
                # to guess the target.
                target_pc = self._resolve_jump_target(block)
                if target_pc is not None and target_pc in pc_to_block_index:
                    target_idx = pc_to_block_index[target_pc]
                    cfg.add_edge(idx, target_idx, type='jump')

        return cfg

    def _resolve_jump_target(self, block):
        # Simple heuristic: look for PUSH before JUMP
        if len(block) >= 2:
            prev = block[-2]
            if prev.name.startswith('PUSH'):
                try:
                    return int(prev.operand, 16) if isinstance(prev.operand, str) else int(prev.operand)
                except:
                    return None
        return None

    def graph_to_data(self, cfg: nx.DiGraph):
        """
        Convert NetworkX CFG to PyTorch Geometric Data object
        """
        from torch_geometric.data import Data

        # 1. Node Features
        # We need to convert block instructions into a fixed-size vector
        # Strategy: Use a sequence of opcode indices (padded/truncated) or Bag-of-Words
        # For GAT, let's use a Bag-of-Opcodes vector (frequency count) for simplicity
        # OR better: Average Embedding.
        
        node_features = []
        node_mapping = {node: i for i, node in enumerate(cfg.nodes())}
        
        for node_id in cfg.nodes():
            block = cfg.nodes[node_id]['instructions']
            feature_vec = self._extract_block_features(block)
            node_features.append(feature_vec)
            
        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Edge Index
        edge_indices = []
        for src, dst in cfg.edges():
            edge_indices.append([node_mapping[src], node_mapping[dst]])
        
        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def _extract_block_features(self, block):
        # Create a frequency vector of opcodes (Bag-of-Opcodes)
        # This is a simple but effective representation for basic blocks
        vec = np.zeros(self.vocab_size)
        
        for instr in block:
            # Map opcode name to index
            name = instr.name
            # Handle PUSH1, PUSH2... as just PUSH if needed, or keep distinct
            # Here we try to match exact name, else hash or ignore
            if name in self.opcode_vocab:
                idx = self.opcode_vocab[name]
            elif name.startswith('PUSH'):
                idx = self.opcode_vocab.get('PUSH1', 64) # Fallback
            elif name.startswith('DUP'):
                idx = self.opcode_vocab.get('DUP1', 96)
            elif name.startswith('SWAP'):
                idx = self.opcode_vocab.get('SWAP1', 112)
            else:
                idx = self.vocab_size - 1 # Unknown
                
            if idx < self.vocab_size:
                vec[idx] += 1
            
        return vec

