# NT547 - CFG Visualizer for Solidity Smart Contracts

D02: Công cụ Trực quan hóa Đồ thị Luồng Điều khiển (CFG) cho Hợp đồng Thông minh

## 📖 Overview

A web-based tool that allows users to paste Solidity source code and receive an interactive, visual representation of the Control Flow Graph (CFG) for functions in the contract. This is an essential tool for security analysts and developers to understand the execution logic of smart contracts.

## ✨ Features

- **Input Interface**: Simple web interface with a code editor for pasting Solidity source code
- **Syntax Parsing**: Uses @solidity-parser/parser to convert source code to Abstract Syntax Tree (AST)
- **CFG Construction**: Logic to traverse the AST, identify functions, basic blocks, and control flow transfer points
- **Visualization**: Graph visualization using React Flow with interactive features
- **Interactive**: Click on graph nodes to highlight corresponding code lines in the editor

## 🏗️ Architecture

The project is structured into two main components:

### Backend (Python Flask)
- Simple REST API for health checks
- Extensible for future server-side processing
- CORS enabled for frontend communication

### Frontend (React)
- Monaco editor for code editing with Solidity syntax highlighting
- @solidity-parser/parser for AST generation
- Custom CFG builder logic
- React Flow for graph visualization
- Interactive node-to-code highlighting

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 14+** (for frontend)
- **npm** or **yarn**

> **Windows Users**: See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed Windows-specific setup instructions.

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will open at `http://localhost:3000`

## 📁 Project Structure

```
NT547/
├── backend/              # Python Flask backend
│   ├── app.py           # Main Flask application
│   ├── requirements.txt # Python dependencies
│   └── README.md        # Backend documentation
│
├── frontend/            # React frontend
│   ├── public/          # Static files
│   ├── src/
│   │   ├── components/  # React components
│   │   │   ├── CodeEditor.js      # Monaco editor wrapper
│   │   │   └── CFGVisualizer.js   # React Flow visualization
│   │   ├── utils/
│   │   │   └── parser.js          # Solidity parser & CFG builder
│   │   ├── App.js       # Main application
│   │   └── index.js     # Entry point
│   ├── package.json
│   └── README.md        # Frontend documentation
│
└── README.md           # This file
```

## 🎯 Usage

1. **Start both backend and frontend servers** (see Quick Start above)
2. **Open the application** in your browser at `http://localhost:3000`
3. **Enter Solidity code** in the left editor panel (or use the provided sample)
4. **Click "Generate CFG"** to create the control flow graph
5. **Interact with the graph**: 
   - Click nodes to highlight corresponding code
   - Zoom and pan the graph
   - Use the minimap for navigation

## 🔧 Technologies Used

### Frontend
- **React 18**: UI framework
- **@solidity-parser/parser**: Solidity AST parsing
- **React Flow**: Graph visualization
- **Monaco Editor**: Code editor with syntax highlighting
- **Axios**: HTTP client

### Backend
- **Flask**: Python web framework
- **Flask-CORS**: CORS support

## 📊 CFG Features

### Node Types
- **Entry/Exit**: Function boundaries
- **Condition**: If/While/For conditions
- **Branch**: True/False paths
- **Merge**: Convergence points
- **Return**: Return statements
- **Statement**: Regular code blocks

### Supported Constructs
- Function definitions
- If/Else statements
- While loops
- For loops
- Return statements
- Variable declarations
- Expression statements
- Nested control structures

## 🖼️ Screenshots

(Screenshots will be added after deployment)

## 🖼️ Screenshots

### Initial Interface
![CFG Visualizer Initial View](https://github.com/user-attachments/assets/af5b4545-45c9-4a2d-a7d6-a5ffb2945c08)

### Control Flow Graph Visualization
![CFG Visualization](https://github.com/user-attachments/assets/a9dc6dce-95be-4822-ae5f-9ce4fa493047)

### Interactive Node Selection
![Interactive Feature](https://github.com/user-attachments/assets/52245b2d-8e4d-4a70-a778-031130d04a94)
*Clicking on nodes highlights corresponding code lines*

## 🛠️ Development

### Running Tests
```bash
cd frontend
npm test
```

### Building for Production
```bash
cd frontend
npm run build
```

## 📝 License

MIT

## 👥 Contributors

UIT - NT547 Course Project

## 🙏 Acknowledgments

- @solidity-parser/parser for Solidity parsing capabilities
- React Flow for excellent graph visualization
- Monaco Editor for the code editing experience