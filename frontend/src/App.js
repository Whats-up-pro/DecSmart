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
  const [securityAnalysis, setSecurityAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showVulnerabilities, setShowVulnerabilities] = useState(false);
  const [highlightedLine, setHighlightedLine] = useState(null);
  const [vulnerabilityPanelHeight, setVulnerabilityPanelHeight] = useState(250);

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

  const handleAnalyzeSecurity = async () => {
    setIsAnalyzing(true);
    setError(null);
    setSecurityAnalysis(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/v1/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setSecurityAnalysis(result);
      setShowVulnerabilities(true);
      
      // Update CFG with vulnerability info if available
      if (result.cfg) {
        setFullCfgData(result.cfg);
        
        // Update contract names
        const names = [];
        const regex = /\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)/g;
        let m;
        while ((m = regex.exec(code)) !== null) {
          const name = m[1];
          if (!names.includes(name)) names.push(name);
        }
        setContractNames(names);
        if (!selectedContractName && names.length > 0) {
          setSelectedContractName(names[0]);
        }
      }
    } catch (err) {
      setError(`Security analysis failed: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleVulnerabilityClick = (vulnerability) => {
    setHighlightedLine(vulnerability.line);
    // Scroll to the line in code editor
    const editorElement = document.querySelector('.code-editor textarea');
    if (editorElement) {
      const lineHeight = 20; // Approximate line height
      const scrollPosition = (vulnerability.line - 1) * lineHeight;
      editorElement.scrollTop = scrollPosition - 100; // Offset for visibility
    }
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
            <button 
              onClick={handleAnalyzeSecurity} 
              className="parse-button analyze-button" 
              disabled={isAnalyzing}
              style={{ marginLeft: 8 }}
            >
              {isAnalyzing ? 'Analyzing...' : '🔍 Security Analysis'}
            </button>
            <button onClick={handleExportJSON} className="parse-button" disabled={!cfgData} style={{ marginLeft: 8 }}>
              Export CFG JSON
            </button>
          </div>
          <CodeEditor 
            value={code} 
            onChange={setCode}
            selectedNode={selectedNode}
            highlightedLine={highlightedLine}
          />
          
          {showVulnerabilities && securityAnalysis && (
            <div 
              className="vulnerability-panel"
              style={{ height: `${vulnerabilityPanelHeight}px` }}
            >
              <div 
                className="resize-handle"
                onMouseDown={(e) => {
                  e.preventDefault();
                  const startY = e.clientY;
                  const startHeight = vulnerabilityPanelHeight;
                  
                  const handleMouseMove = (e) => {
                    const deltaY = startY - e.clientY;
                    const newHeight = Math.max(150, Math.min(600, startHeight + deltaY));
                    setVulnerabilityPanelHeight(newHeight);
                  };
                  
                  const handleMouseUp = () => {
                    document.removeEventListener('mousemove', handleMouseMove);
                    document.removeEventListener('mouseup', handleMouseUp);
                  };
                  
                  document.addEventListener('mousemove', handleMouseMove);
                  document.addEventListener('mouseup', handleMouseUp);
                }}
              >
                <div className="resize-handle-bar"></div>
              </div>
              
              <div className="vulnerability-content">
                <div className="vulnerability-header">
                  <h3>Security Analysis Results</h3>
                  <button 
                    onClick={() => {
                      setShowVulnerabilities(false);
                      setHighlightedLine(null);
                    }} 
                    className="close-button"
                    style={{ background: 'transparent', border: 'none', fontSize: '20px', cursor: 'pointer', color: '#999' }}
                  >
                    ×
                  </button>
                </div>
              
              <div className="security-score">
                <div className="score-label">Security Score:</div>
                <div className={`score-value ${securityAnalysis.score >= 80 ? 'good' : securityAnalysis.score >= 50 ? 'medium' : 'bad'}`}>
                  {securityAnalysis.score}/100
                </div>
              </div>
              
              <div className="vulnerability-summary">
                <h4>Summary</h4>
                <div className="summary-stats">
                  <span className="stat critical">Critical: {securityAnalysis.summary.bySeverity.critical}</span>
                  <span className="stat high">High: {securityAnalysis.summary.bySeverity.high}</span>
                  <span className="stat medium">Medium: {securityAnalysis.summary.bySeverity.medium}</span>
                  <span className="stat low">Low: {securityAnalysis.summary.bySeverity.low}</span>
                  <span className="stat info">Info: {securityAnalysis.summary.bySeverity.info}</span>
                </div>
              </div>
              
              <div className="vulnerability-list">
                <h4>Vulnerabilities ({securityAnalysis.vulnerabilities.length})</h4>
                {securityAnalysis.vulnerabilities.length === 0 ? (
                  <p className="no-vulnerabilities">✅ No vulnerabilities detected</p>
                ) : (
                  <div className="vulnerabilities">
                    {securityAnalysis.vulnerabilities.map((vuln, idx) => (
                      <div 
                        key={idx} 
                        className={`vulnerability-item ${vuln.severity}`}
                        onClick={() => handleVulnerabilityClick(vuln)}
                        style={{ cursor: 'pointer' }}
                      >
                        <div className="vuln-header">
                          <span className="vuln-type">{vuln.type}</span>
                          <span className={`vuln-severity ${vuln.severity}`}>{vuln.severity.toUpperCase()}</span>
                        </div>
                        <div className="vuln-line">Line {vuln.line}</div>
                        <div className="vuln-description">{vuln.description}</div>
                        <div className="vuln-recommendation">
                          <strong>💡 Recommendation:</strong> {vuln.recommendation}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              </div>
            </div>
          )}
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
