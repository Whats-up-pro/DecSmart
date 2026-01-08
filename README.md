# DecSmart - Smart Contract Vulnerability Detection

**CFG Visualizer & Security Analyzer for Solidity Smart Contracts**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-14+-green.svg)](https://nodejs.org)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com)

## 📖 Overview

A web-based tool for **smart contract security analysis** that combines:
- **Control Flow Graph (CFG)** visualization
- **AI-powered vulnerability detection** using HiFi-GAT GNN model
- **Pattern-based security analysis** for common vulnerabilities

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                               │
│                        (React - Port 3000)                           │
├─────────────────┬─────────────────────────┬─────────────────────────┤
│   CodeEditor    │    CFGVisualizer        │   VulnerabilityPanel    │
│   (Monaco)      │    (ReactFlow)          │   (Results Display)     │
└────────┬────────┴────────────┬────────────┴────────────┬────────────┘
         │                     │                         │
         │         POST /api/v1/analyze                  │
         │         POST /api/v1/cfg                      │
         └─────────────────────┼─────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         BACKEND LAYER                                │
│                       (Flask - Port 5000)                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  Flask API  │───▶│ SecurityAnalyzer │───▶│  HiFi-GAT Model   │   │
│  │  Gateway    │    │                  │    │  (GNN Inference)  │   │
│  └─────────────┘    │  - Regex Pattern │    └───────────────────┘   │
│                     │  - CFG Analysis  │                             │
│  ┌─────────────┐    │  - GNN Inference │    ┌───────────────────┐   │
│  │ node_helper │    └──────────────────┘    │    CFGBuilder     │   │
│  │ (Solidity   │                            │  (EVM Bytecode →  │   │
│  │  Parser)    │                            │   HiFi-CFG)       │   │
│  └─────────────┘                            └───────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
NT547/
├── backend/                    # Python Flask Backend
│   ├── app.py                  # Flask app factory
│   ├── api/
│   │   ├── __init__.py         # Blueprint registration
│   │   └── routes.py           # API endpoints (/cfg, /analyze)
│   ├── model/
│   │   ├── gnn.py              # HiFi-GAT neural network model
│   │   ├── preprocess.py       # CFGBuilder (bytecode → graph)
│   │   ├── train.py            # Model training script
│   │   └── dataset.py          # Dataset loader
│   ├── security_analyzer.py    # Vulnerability detection engine
│   ├── node_helper/            # Node.js Solidity parser
│   │   └── index.js            # AST/CFG generation
│   ├── saved_models/
│   │   └── hifi_gat.pth        # Pre-trained GNN model
│   └── requirements.txt
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.js              # Main application
│   │   ├── components/
│   │   │   ├── CodeEditor.js   # Monaco editor
│   │   │   └── CFGVisualizer.js# ReactFlow graph
│   │   └── utils/
│   │       └── parser.js       # Client-side Solidity parser
│   └── package.json
│
└── README.md
```

---

## 🔧 Technology Stack

### Backend Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | Flask 3.0 | REST API server |
| **GNN Model** | PyTorch + PyG | HiFi-GAT vulnerability detection |
| **Bytecode Analysis** | pyevmasm | EVM disassembly |
| **Solidity Compiler** | py-solc-x | Source → Bytecode |
| **Parser** | Node.js + @solidity-parser | AST/CFG generation |

### Frontend Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | React 18 | Component-based UI |
| **Code Editor** | Monaco Editor | Syntax highlighting, line navigation |
| **Graph Visualization** | ReactFlow 11 | Interactive CFG display |
| **HTTP Client** | Axios | API communication |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js 14+**
- **npm**

### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Install Node helper
cd node_helper && npm ci && cd ..

# Start server
python app.py
```

Backend runs on `http://localhost:5000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend opens at `http://localhost:3000`

---

## � API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/cfg` | POST | Build CFG from Solidity code |
| `/api/v1/analyze` | POST | Security analysis (CFG + Vulnerabilities) |

### Example Request

```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "pragma solidity ^0.8.0; contract Test { ... }"}'
```

### Response Format

```json
{
  "vulnerabilities": [
    {
      "type": "Reentrancy",
      "severity": "critical",
      "line": 15,
      "description": "...",
      "recommendation": "..."
    }
  ],
  "cfg": { "nodes": [...], "edges": [...] },
  "score": 75,
  "summary": { "bySeverity": {...}, "byType": {...} }
}
```

---

## 🛡️ Vulnerability Detection

### Detection Methods

1. **Regex Pattern Matching**: Fast detection of common patterns
2. **CFG Analysis**: Unreachable code, infinite loops
3. **HiFi-GAT GNN**: AI-based detection on bytecode CFG

### Supported Vulnerabilities

| Type | Severity | Detection Method |
|------|----------|------------------|
| Reentrancy | Critical | Regex + GNN |
| Unprotected Selfdestruct | Critical | Regex |
| Unchecked Call Return | High | Regex |
| Delegatecall | High | Regex |
| Integer Overflow | High | Regex |
| tx.origin Auth | Medium | Regex |
| Timestamp Dependence | Medium | Regex |
| DoS with Gas Limit | Medium | Regex + CFG |
| Missing Access Control | High | Regex |
| Unreachable Code | Info | CFG |

---

## � Screenshots

### Initial Interface
![CFG Visualizer Initial View](https://github.com/user-attachments/assets/af5b4545-45c9-4a2d-a7d6-a5ffb2945c08)

### Control Flow Graph Visualization
![CFG Visualization](https://github.com/user-attachments/assets/a9dc6dce-95be-4822-ae5f-9ce4fa493047)

### Interactive Node Selection
![Interactive Feature](https://github.com/user-attachments/assets/52245b2d-8e4d-4a70-a778-031130d04a94)

---

## 📝 License

MIT