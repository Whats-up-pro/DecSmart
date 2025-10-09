import React, { useEffect, useRef, useState } from 'react';
import './CodeEditor.css';

function CodeEditor({ value, onChange, selectedNode }) {
  const [cursorPos, setCursorPos] = useState({ line: 1, col: 1 });
  const textareaRef = useRef(null);

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

  // When a graph node is selected, highlight its corresponding code lines
  useEffect(() => {
    if (!selectedNode || !textareaRef.current || !value) return;

    const startLine = Math.max(1, selectedNode.startLine || 1);
    const endLine = Math.max(startLine, selectedNode.endLine || startLine);

    // Compute character offsets for start of startLine and end of endLine
    const getLineStartOffset = (text, lineNumber) => {
      if (lineNumber <= 1) return 0;
      let count = 1;
      for (let i = 0; i < text.length; i++) {
        if (text[i] === '\n') {
          count += 1;
          if (count === lineNumber) {
            return i + 1; // char after the newline
          }
        }
      }
      return text.length;
    };

    const getLineEndOffset = (text, lineNumber) => {
      // End offset is position of newline at endLine or text end
      if (lineNumber < 1) return 0;
      let count = 1;
      for (let i = 0; i < text.length; i++) {
        if (text[i] === '\n') {
          if (count === lineNumber) return i; // newline index (exclusive)
          count += 1;
        }
      }
      // If file ends without a trailing newline and endLine is last line
      return text.length;
    };

    const start = getLineStartOffset(value, startLine);
    const end = getLineEndOffset(value, endLine);
    const el = textareaRef.current;

    // Apply selection and scroll into view
    try {
      el.focus();
      el.setSelectionRange(start, end);
    } catch (_) {
      // ignore browsers that might not support setSelectionRange on some inputs
    }
  }, [selectedNode, value]);

  return (
    <div className="code-editor">
      <textarea
        ref={textareaRef}
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
