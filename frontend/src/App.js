import React, { useState } from 'react';
import './App.css';
import CodeEditor from './components/CodeEditor';
import CFGVisualizer from './components/CFGVisualizer';
import { parseSolidityCode } from './utils/parser';

const SAMPLE_CODE = `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleContract {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        if (_value > 100) {
            value = 100;
        } else {
            value = _value;
        }
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}`;

function App() {
  const [code, setCode] = useState(SAMPLE_CODE);
  const [cfgData, setCfgData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  const handleParse = () => {
    try {
      setError(null);
      const result = parseSolidityCode(code);
      setCfgData(result);
    } catch (err) {
      setError(err.message);
      setCfgData(null);
    }
  };

  const handleNodeClick = (nodeData) => {
    setSelectedNode(nodeData);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>CFG Visualizer for Solidity Smart Contracts</h1>
        <p>Control Flow Graph Visualization Tool</p>
      </header>
      
      <div className="container">
        <div className="editor-panel">
          <div className="panel-header">
            <h2>Solidity Code</h2>
            <button onClick={handleParse} className="parse-button">
              Generate CFG
            </button>
          </div>
          <CodeEditor 
            value={code} 
            onChange={setCode}
            selectedNode={selectedNode}
          />
        </div>
        
        <div className="visualizer-panel">
          <div className="panel-header">
            <h2>Control Flow Graph</h2>
          </div>
          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}
          <CFGVisualizer 
            cfgData={cfgData}
            onNodeClick={handleNodeClick}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
