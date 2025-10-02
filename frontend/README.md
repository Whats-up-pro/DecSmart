# CFG Visualizer Frontend

A React-based frontend for visualizing Control Flow Graphs (CFG) of Solidity smart contracts.

## Features

- 🎨 **Code Editor**: Monaco editor with Solidity syntax highlighting
- 📊 **CFG Visualization**: Interactive graph visualization using React Flow
- 🔍 **Interactive Navigation**: Click on CFG nodes to highlight corresponding code
- 🎯 **AST Parsing**: Uses @solidity-parser/parser for accurate Solidity parsing
- 📱 **Responsive Design**: Works on desktop and tablet devices

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn

## Installation

1. Install dependencies:
```bash
npm install
```

## Running the Application

Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000`

## Building for Production

Create a production build:
```bash
npm run build
```

The build artifacts will be stored in the `build/` directory.

## Usage

1. Paste your Solidity smart contract code in the editor (or use the sample code provided)
2. Click the "Generate CFG" button
3. View the generated Control Flow Graph in the right panel
4. Click on any node in the graph to highlight the corresponding code lines in the editor

## Technologies Used

- **React**: UI framework
- **@solidity-parser/parser**: Solidity AST parser
- **React Flow**: Graph visualization library
- **Monaco Editor**: Code editor component
- **Axios**: HTTP client for API calls

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── CodeEditor.js       # Monaco editor wrapper
│   │   ├── CodeEditor.css
│   │   ├── CFGVisualizer.js    # React Flow CFG visualization
│   │   └── CFGVisualizer.css
│   ├── utils/
│   │   └── parser.js           # Solidity parser and CFG builder
│   ├── App.js                  # Main application component
│   ├── App.css
│   ├── index.js                # Application entry point
│   └── index.css
└── package.json
```

## CFG Node Types

- **Entry Node**: Function entry point (blue background)
- **Exit Node**: Function exit point (pink background)
- **Condition Node**: If/While/For conditions (green background)
- **Return Node**: Return statements (orange background)
- **Regular Node**: Standard statements (white background)
- **Merge Node**: Convergence point after branches

## Supported Solidity Constructs

- Function definitions
- If/Else statements
- While loops
- For loops
- Return statements
- Variable declarations
- Expression statements
- Function calls
