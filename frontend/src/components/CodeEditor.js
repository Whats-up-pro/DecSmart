import React from 'react';
import './CodeEditor.css';

function CodeEditor({ value, onChange, selectedNode }) {
  const handleChange = (e) => {
    onChange(e.target.value);
  };

  return (
    <div className="code-editor">
      <textarea
        value={value}
        onChange={handleChange}
        className="code-textarea"
        spellCheck="false"
      />
      {selectedNode && (
        <div className="selected-info">
          Selected: Lines {selectedNode.startLine} - {selectedNode.endLine}
        </div>
      )}
    </div>
  );
}

export default CodeEditor;
