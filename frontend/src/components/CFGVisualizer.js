import React, { useCallback, useMemo, useEffect, useRef } from 'react';
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
  const containerRef = useRef(null);
  const rfInstanceRef = useRef(null);
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

  const onMoveEnd = useCallback((_, viewport) => {
    const containerEl = containerRef.current;
    if (!viewport || !containerEl) return;

    const { x, y, zoom } = viewport;
    const width = containerEl.clientWidth || 0;
    const height = containerEl.clientHeight || 0;

    if (width === 0 || height === 0 || !zoom) return;

    // Visible rectangle in flow coordinates
    const left = -x / zoom;
    const top = -y / zoom;
    const right = (-x + width) / zoom;
    const bottom = (-y + height) / zoom;

    const nodesList = rfInstanceRef.current?.getNodes?.() || [];
    const nodesInView = nodesList.filter((n) => {
      if (n == null) return false;
      const nodeWidth = n.width ?? 0;
      const nodeHeight = n.height ?? 0;
      const nx1 = n.position?.x ?? 0;
      const ny1 = n.position?.y ?? 0;
      const nx2 = nx1 + nodeWidth;
      const ny2 = ny1 + nodeHeight;
      // AABB intersect
      return !(nx2 < left || nx1 > right || ny2 < top || ny1 > bottom);
    });

    if (nodesInView.length === 0) return;

    const { sumX, sumY } = nodesInView.reduce(
      (acc, n) => {
        const nodeWidth = n.width ?? 0;
        const nodeHeight = n.height ?? 0;
        const cx = (n.position?.x ?? 0) + nodeWidth / 2;
        const cy = (n.position?.y ?? 0) + nodeHeight / 2;
        acc.sumX += cx;
        acc.sumY += cy;
        return acc;
      },
      { sumX: 0, sumY: 0 }
    );

    const centerX = sumX / nodesInView.length;
    const centerY = sumY / nodesInView.length;

    // Offset so selected region appears slightly toward top-left in the main view
    const offsetPxRatio = 0.06; // 6% of viewport
    const offsetXFlow = (width * offsetPxRatio) / zoom;
    const offsetYFlow = (height * offsetPxRatio) / zoom;

    // Recenter on centroid while preserving zoom
    rfInstanceRef.current?.setCenter?.(centerX - offsetXFlow, centerY - offsetYFlow, { zoom, duration: 0 });
  }, []);

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
    <div className="cfg-visualizer" ref={containerRef}>
      <ReactFlow
        onInit={(instance) => { rfInstanceRef.current = instance; }}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClickHandler}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
        onMoveEnd={onMoveEnd}
      >
        <Controls />
        <MiniMap 
          nodeStrokeColor={(n) => {
            const border = n?.style?.border;
            if (typeof border === 'string' && border.trim()) {
              const parts = border.split(' ');
              return parts[parts.length - 1];
            }
            if (n.type === 'entry') return '#0041d0';
            if (n.type === 'exit') return '#ff0072';
            return '#666';
          }}
          nodeColor={(n) => {
            if (n?.style?.background) return n.style.background;
            if (n.type === 'entry') return '#e3f2fd';
            if (n.type === 'exit') return '#fce4ec';
            return '#fff';
          }}
          pannable
          zoomable
          style={{ width: 160, height: 110, background: '#000' }}
          viewportStyle={{ stroke: '#00e676', strokeWidth: 4, fill: 'rgba(0,230,118,0.14)' }}
          className="cfg-minimap"
        />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
}

export default CFGVisualizer;
