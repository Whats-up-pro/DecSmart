import React, { useCallback, useMemo, useEffect } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './CFGVisualizer.css';

const nodeTypes = {
  // Can add custom node types here if needed
};

function CFGVisualizer({ cfgData, onNodeClick }) {
  const initialNodes = useMemo(() => {
    if (!cfgData || !cfgData.nodes) return [];
    return cfgData.nodes;
  }, [cfgData]);

  const initialEdges = useMemo(() => {
    if (!cfgData || !cfgData.edges) return [];
    return cfgData.edges;
  }, [cfgData]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes and edges when cfgData changes
  useEffect(() => {
    if (cfgData) {
      setNodes(cfgData.nodes || []);
      setEdges(cfgData.edges || []);
    }
  }, [cfgData, setNodes, setEdges]);

  const onNodeClickHandler = useCallback(
    (event, node) => {
      if (onNodeClick) {
        onNodeClick(node.data);
      }
    },
    [onNodeClick]
  );

  if (!cfgData) {
    return (
      <div className="cfg-placeholder">
        <div className="placeholder-content">
          <svg
            width="120"
            height="120"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="12" cy="5" r="3" />
            <circle cx="12" cy="19" r="3" />
            <circle cx="5" cy="12" r="3" />
            <circle cx="19" cy="12" r="3" />
            <line x1="12" y1="8" x2="12" y2="16" />
            <line x1="8" y1="12" x2="16" y2="12" />
          </svg>
          <p>Enter Solidity code and click "Generate CFG" to visualize the control flow graph</p>
        </div>
      </div>
    );
  }

  return (
    <div className="cfg-visualizer">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClickHandler}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Controls />
        <MiniMap 
          nodeStrokeColor={(n) => {
            if (n.type === 'entry') return '#0041d0';
            if (n.type === 'exit') return '#ff0072';
            return '#666';
          }}
          nodeColor={(n) => {
            if (n.type === 'entry') return '#e3f2fd';
            if (n.type === 'exit') return '#fce4ec';
            return '#fff';
          }}
        />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export default CFGVisualizer;
