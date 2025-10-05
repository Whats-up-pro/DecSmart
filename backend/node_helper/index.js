import parser from '@solidity-parser/parser';

let nodeIdCounter = 0;
function getNodeId() {
  return `node_${nodeIdCounter++}`;
}
function resetNodeId() {
  nodeIdCounter = 0;
}

function getStatementLabel(stmt) {
  switch (stmt.type) {
    case 'ExpressionStatement':
      return 'Expression';
    case 'VariableDeclarationStatement':
      return 'Variable declaration';
    case 'EmitStatement':
      return 'Emit event';
    case 'FunctionCall':
      return 'Function call';
    default:
      return stmt.type;
  }
}

function processStatements(statements, prevNodeId, exitId, xPosition, yPosition, nodes, edges) {
  let currentNodeId = prevNodeId;
  let currentY = yPosition;
  for (const stmt of statements) {
    const result = processStatement(stmt, currentNodeId, exitId, xPosition, currentY, nodes, edges);
    currentNodeId = result.lastNodeId;
    currentY = result.yPosition;
  }
  return { lastNodeId: currentNodeId, yPosition: currentY };
}

function processIfStatement(stmt, prevNodeId, exitId, xPosition, yPosition, nodes, edges) {
  const conditionNodeId = getNodeId();
  nodes.push({
    id: conditionNodeId,
    type: 'default',
    data: {
      label: 'If condition',
      startLine: stmt.loc?.start.line,
      endLine: stmt.loc?.start.line,
    },
    position: { x: xPosition, y: yPosition },
    style: { background: '#4caf50', border: '2px solid #388e3c', color: 'white' },
  });
  edges.push({ id: `${prevNodeId}-${conditionNodeId}`, source: prevNodeId, target: conditionNodeId, type: 'smoothstep' });

  const trueBranchId = getNodeId();
  const trueY = yPosition + 100;
  nodes.push({
    id: trueBranchId,
    type: 'default',
    data: {
      label: 'True',
      startLine: stmt.trueBody?.loc?.start.line,
      endLine: stmt.trueBody?.loc?.start.line,
    },
    position: { x: xPosition - 150, y: trueY },
    style: { background: '#e8f5e9' },
  });
  edges.push({ id: `${conditionNodeId}-${trueBranchId}`, source: conditionNodeId, target: trueBranchId, type: 'smoothstep', label: 'true', style: { stroke: '#4caf50' } });

  let trueEndNodeId = trueBranchId;
  if (stmt.trueBody) {
    const trueStatements = stmt.trueBody.type === 'Block' ? stmt.trueBody.statements : [stmt.trueBody];
    const result = processStatements(trueStatements, trueBranchId, exitId, xPosition - 150, trueY + 100, nodes, edges);
    trueEndNodeId = result.lastNodeId;
  }

  let falseEndNodeId;
  let maxY = trueY + 200;

  if (stmt.falseBody) {
    const falseBranchId = getNodeId();
    const falseY = yPosition + 100;
    nodes.push({
      id: falseBranchId,
      type: 'default',
      data: {
        label: 'False',
        startLine: stmt.falseBody?.loc?.start.line,
        endLine: stmt.falseBody?.loc?.start.line,
      },
      position: { x: xPosition + 150, y: falseY },
      style: { background: '#ffebee' },
    });
    edges.push({ id: `${conditionNodeId}-${falseBranchId}`, source: conditionNodeId, target: falseBranchId, type: 'smoothstep', label: 'false', style: { stroke: '#f44336' } });

    const falseStatements = stmt.falseBody.type === 'Block' ? stmt.falseBody.statements : [stmt.falseBody];
    const result = processStatements(falseStatements, falseBranchId, exitId, xPosition + 150, falseY + 100, nodes, edges);
    falseEndNodeId = result.lastNodeId;
    maxY = Math.max(maxY, result.yPosition);
  } else {
    falseEndNodeId = conditionNodeId;
    edges.push({ id: `${conditionNodeId}-merge-false`, source: conditionNodeId, target: conditionNodeId, type: 'smoothstep', label: 'false', style: { stroke: '#f44336' }, hidden: true });
  }

  const mergeNodeId = getNodeId();
  nodes.push({ id: mergeNodeId, type: 'default', data: { label: 'Merge', startLine: stmt.loc?.end.line, endLine: stmt.loc?.end.line }, position: { x: xPosition, y: maxY } });
  edges.push({ id: `${trueEndNodeId}-${mergeNodeId}`, source: trueEndNodeId, target: mergeNodeId, type: 'smoothstep' });
  if (stmt.falseBody) {
    edges.push({ id: `${falseEndNodeId}-${mergeNodeId}`, source: falseEndNodeId, target: mergeNodeId, type: 'smoothstep' });
  } else {
    edges.push({ id: `${conditionNodeId}-${mergeNodeId}`, source: conditionNodeId, target: mergeNodeId, type: 'smoothstep', label: 'false' });
  }
  return { lastNodeId: mergeNodeId, yPosition: maxY + 100 };
}

function processLoopStatement(stmt, prevNodeId, exitId, xPosition, yPosition, nodes, edges) {
  const conditionNodeId = getNodeId();
  nodes.push({
    id: conditionNodeId,
    type: 'default',
    data: { label: `${stmt.type === 'WhileStatement' ? 'While' : 'For'} condition`, startLine: stmt.loc?.start.line, endLine: stmt.loc?.start.line },
    position: { x: xPosition, y: yPosition },
    style: { background: '#2196f3', border: '2px solid #1976d2', color: 'white' },
  });
  edges.push({ id: `${prevNodeId}-${conditionNodeId}`, source: prevNodeId, target: conditionNodeId, type: 'smoothstep' });

  const bodyNodeId = getNodeId();
  const bodyY = yPosition + 100;
  nodes.push({ id: bodyNodeId, type: 'default', data: { label: 'Loop body', startLine: stmt.body?.loc?.start.line, endLine: stmt.body?.loc?.end.line }, position: { x: xPosition - 100, y: bodyY } });
  edges.push({ id: `${conditionNodeId}-${bodyNodeId}`, source: conditionNodeId, target: bodyNodeId, type: 'smoothstep', label: 'true', style: { stroke: '#4caf50' } });

  let bodyEndNodeId = bodyNodeId;
  if (stmt.body) {
    const bodyStatements = stmt.body.type === 'Block' ? stmt.body.statements : [stmt.body];
    const result = processStatements(bodyStatements, bodyNodeId, exitId, xPosition - 100, bodyY + 100, nodes, edges);
    bodyEndNodeId = result.lastNodeId;
  }

  edges.push({ id: `${bodyEndNodeId}-${conditionNodeId}`, source: bodyEndNodeId, target: conditionNodeId, type: 'smoothstep', animated: true });

  const exitLoopNodeId = getNodeId();
  nodes.push({ id: exitLoopNodeId, type: 'default', data: { label: 'Exit loop', startLine: stmt.loc?.end.line, endLine: stmt.loc?.end.line }, position: { x: xPosition + 100, y: yPosition + 100 } });
  edges.push({ id: `${conditionNodeId}-${exitLoopNodeId}`, source: conditionNodeId, target: exitLoopNodeId, type: 'smoothstep', label: 'false', style: { stroke: '#f44336' } });

  return { lastNodeId: exitLoopNodeId, yPosition: bodyY + 200 };
}

function processStatement(stmt, prevNodeId, exitId, xPosition, yPosition, nodes, edges) {
  switch (stmt.type) {
    case 'IfStatement':
      return processIfStatement(stmt, prevNodeId, exitId, xPosition, yPosition, nodes, edges);
    case 'WhileStatement':
    case 'ForStatement':
      return processLoopStatement(stmt, prevNodeId, exitId, xPosition, yPosition, nodes, edges);
    case 'ReturnStatement': {
      const returnNodeId = getNodeId();
      nodes.push({ id: returnNodeId, type: 'default', data: { label: 'Return', startLine: stmt.loc?.start.line, endLine: stmt.loc?.end.line }, position: { x: xPosition, y: yPosition }, style: { background: '#ffc107', border: '2px solid #ff9800' } });
      edges.push({ id: `${prevNodeId}-${returnNodeId}`, source: prevNodeId, target: returnNodeId, type: 'smoothstep' });
      edges.push({ id: `${returnNodeId}-${exitId}`, source: returnNodeId, target: exitId, type: 'smoothstep' });
      return { lastNodeId: returnNodeId, yPosition: yPosition + 100 };
    }
    default: {
      const nodeId = getNodeId();
      const label = getStatementLabel(stmt);
      nodes.push({ id: nodeId, type: 'default', data: { label, startLine: stmt.loc?.start.line, endLine: stmt.loc?.end.line }, position: { x: xPosition, y: yPosition } });
      edges.push({ id: `${prevNodeId}-${nodeId}`, source: prevNodeId, target: nodeId, type: 'smoothstep' });
      return { lastNodeId: nodeId, yPosition: yPosition + 100 };
    }
  }
}

function buildCFGForFunction(functionNode, functionName) {
  const nodes = [];
  const edges = [];
  const entryId = getNodeId();
  const exitId = getNodeId();
  nodes.push({ id: entryId, type: 'entry', data: { label: `Entry: ${functionName}`, startLine: functionNode.loc?.start.line, endLine: functionNode.loc?.start.line }, position: { x: 200, y: 0 } });
  let currentNodeId = entryId;
  let yPosition = 100;
  if (functionNode.body && functionNode.body.statements) {
    const statementResult = processStatements(functionNode.body.statements, currentNodeId, exitId, 200, yPosition, nodes, edges);
    currentNodeId = statementResult.lastNodeId;
    yPosition = statementResult.yPosition;
  }
  nodes.push({ id: exitId, type: 'exit', data: { label: `Exit: ${functionName}`, startLine: functionNode.loc?.end.line, endLine: functionNode.loc?.end.line }, position: { x: 200, y: yPosition } });
  if (currentNodeId !== exitId) {
    edges.push({ id: `${currentNodeId}-${exitId}`, source: currentNodeId, target: exitId, type: 'smoothstep' });
  }
  return { nodes, edges };
}

function parseSolidityCode(code) {
  resetNodeId();
  const ast = parser.parse(code, { loc: true, range: true });
  const nodes = [];
  const edges = [];
  const functionGraphs = [];
  parser.visit(ast, {
    FunctionDefinition(node) {
      const functionName = node.name || 'fallback';
      const cfg = buildCFGForFunction(node, functionName);
      if (cfg.nodes.length > 0) {
        functionGraphs.push({ name: functionName, ...cfg });
      }
    },
  });

  let yOffset = 0;
  functionGraphs.forEach((funcGraph, index) => {
    const xOffset = index * 400;
    const labelId = getNodeId();
    nodes.push({
      id: labelId,
      type: 'default',
      data: { label: `Function: ${funcGraph.name}`, startLine: 1, endLine: 1 },
      position: { x: xOffset + 150, y: yOffset },
      style: { background: '#667eea', color: 'white', border: '2px solid #764ba2', fontWeight: 'bold', padding: '10px' },
    });
    funcGraph.nodes.forEach((node) => {
      nodes.push({ ...node, position: { x: node.position.x + xOffset, y: node.position.y + yOffset + 80 } });
    });
    edges.push(...funcGraph.edges);
    yOffset += 400;
  });
  return { nodes, edges };
}

async function main() {
  try {
    const input = await new Promise((resolve, reject) => {
      let data = '';
      process.stdin.setEncoding('utf8');
      process.stdin.on('data', (chunk) => (data += chunk));
      process.stdin.on('end', () => resolve(data));
      process.stdin.on('error', reject);
    });
    const payload = JSON.parse(input || '{}');
    const code = payload.code || '';
    if (!code) {
      console.log(JSON.stringify({ error: "Missing 'code'" }));
      process.exit(0);
      return;
    }
    const result = parseSolidityCode(code);
    console.log(JSON.stringify(result));
  } catch (err) {
    console.error(String(err && err.message ? err.message : err));
    process.exit(1);
  }
}

main();


