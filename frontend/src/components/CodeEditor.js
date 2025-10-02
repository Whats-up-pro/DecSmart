import React, { useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import './CodeEditor.css';

function CodeEditor({ value, onChange, selectedNode }) {
  const editorRef = useRef(null);
  const decorationsRef = useRef([]);

  useEffect(() => {
    if (editorRef.current && selectedNode) {
      const { startLine, endLine } = selectedNode;
      
      // Clear previous decorations
      decorationsRef.current = editorRef.current.deltaDecorations(
        decorationsRef.current,
        []
      );

      // Add new decorations
      if (startLine && endLine) {
        decorationsRef.current = editorRef.current.deltaDecorations(
          [],
          [
            {
              range: {
                startLineNumber: startLine,
                startColumn: 1,
                endLineNumber: endLine,
                endColumn: 1,
              },
              options: {
                isWholeLine: true,
                className: 'highlighted-line',
                glyphMarginClassName: 'highlighted-glyph',
              },
            },
          ]
        );

        // Scroll to the highlighted line
        editorRef.current.revealLineInCenter(startLine);
      }
    }
  }, [selectedNode]);

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor;
  };

  return (
    <div className="code-editor">
      <Editor
        height="100%"
        defaultLanguage="sol"
        value={value}
        onChange={onChange}
        onMount={handleEditorDidMount}
        theme="vs-dark"
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 2,
        }}
      />
    </div>
  );
}

export default CodeEditor;
