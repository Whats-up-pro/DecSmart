import React, { useState } from 'react';
import './CodeEditor.css';

function CodeEditor({ value, onChange, selectedNode }) {
  const [cursorPos, setCursorPos] = useState({ line: 1, col: 1 });

  const updateCursorFromEvent = (e) => {
    const pos = e.target.selectionStart || 0;
    const textUpToPos = e.target.value.slice(0, pos);
    const line = textUpToPos.split('\n').length; // 1-based
    const lastNewlineIndex = textUpToPos.lastIndexOf('\n');
    const col = pos - (lastNewlineIndex + 1) + 1; // 1-based
    setCursorPos({ line, col });
  };

  const handleChange = (e) => {
    onChange(e.target.value);
    updateCursorFromEvent(e);
  };

  return (
    <div className="code-editor">
      <textarea
        value={value}
        onChange={handleChange}
        onSelect={updateCursorFromEvent}
        onKeyUp={updateCursorFromEvent}
        onClick={updateCursorFromEvent}
        className="code-textarea"
        spellCheck="false"
      />
      <div className="selected-info">
        <div>Current line:  {cursorPos.line}</div>
        {selectedNode && (
          <div>Selected: Lines {selectedNode.startLine} - {selectedNode.endLine}</div>
        )}
      </div>
    </div>
  );
}

export default CodeEditor;
