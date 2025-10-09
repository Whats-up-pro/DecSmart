import React, { useState, useEffect } from 'react';
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
  const [fullCfgData, setFullCfgData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [contractNames, setContractNames] = useState([]);
  const [selectedContractName, setSelectedContractName] = useState('');

  const handleParse = () => {
    try {
      setError(null);
      const result = parseSolidityCode(code);
      setFullCfgData(result);

      // Detect contracts by simple regex (keeps original parser untouched)
      const names = [];
      const regex = /\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)/g;
      let m;
      while ((m = regex.exec(code)) !== null) {
        const name = m[1];
        if (!names.includes(name)) names.push(name);
      }
      setContractNames(names);
      setSelectedContractName(names[0] || '');
    } catch (err) {
      setError(err.message);
      setCfgData(null);
      setFullCfgData(null);
      setContractNames([]);
      setSelectedContractName('');
    }
  };
  // derive filtered CFG by selected contract
  const filteredCfg = React.useMemo(() => {
    if (!fullCfgData) return null;
    if (!selectedContractName) return fullCfgData;
    const nodes = (fullCfgData.nodes || []).filter(n => n?.data?.contractName === selectedContractName);
    const nodeIds = new Set(nodes.map(n => n.id));
    const edges = (fullCfgData.edges || []).filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    return { nodes, edges };
  }, [fullCfgData, selectedContractName]);

  useEffect(() => {
    setCfgData(filteredCfg);
  }, [filteredCfg]);


  const handleExportJSON = () => {
    if (!cfgData) return;
    const payload = {
      version: 1,
      generatedAt: new Date().toISOString(),
      sourceLanguage: 'solidity',
      source: code,
      cfg: cfgData,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cfg.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
            <button onClick={handleExportJSON} className="parse-button" disabled={!cfgData} style={{ marginLeft: 8 }}>
              Export CFG JSON
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
            {contractNames.length > 1 && (
              <select
                value={selectedContractName}
                onChange={(e) => setSelectedContractName(e.target.value)}
                className="contract-select"
                style={{ minWidth: 180 }}
              >
                {contractNames.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            )}
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
