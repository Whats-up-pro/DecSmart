from flask import request, jsonify
from . import api_bp
import json
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from security_analyzer import SecurityAnalyzer


def _node_helper_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    node_helper_dir = os.path.join(project_root, 'node_helper')
    return node_helper_dir


@api_bp.post('/cfg')
def build_cfg():
    try:
        data = request.get_json(force=True) or {}
        code = data.get('code', '')
        if not code or not isinstance(code, str):
            return jsonify({"error": "Missing 'code' (Solidity source) in request body"}), 400

        node_dir = _node_helper_path()
        js_entry = os.path.join(node_dir, 'index.js')
        if not os.path.exists(js_entry):
            return jsonify({"error": "Node helper not installed. Run 'cd backend/node_helper && npm ci'"}), 500

        # Spawn node to process the code via stdin to avoid CLI length limits
        proc = subprocess.Popen(
            ['node', js_entry],
            cwd=node_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        input_payload = json.dumps({"code": code})
        stdout, stderr = proc.communicate(input=input_payload, timeout=30)
        if proc.returncode != 0:
            return jsonify({"error": "Node helper failed", "details": stderr.strip()}), 500

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON from node helper", "raw": stdout[:500]}), 500

        return jsonify(result)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Node helper timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.post('/analyze')
def analyze_security():
    """
    Analyze Solidity code for security vulnerabilities
    Combines AST/CFG analysis with pattern matching
    """
    try:
        data = request.get_json(force=True) or {}
        code = data.get('code', '')
        
        if not code or not isinstance(code, str):
            return jsonify({"error": "Missing 'code' in request body"}), 400
        
        # First, get CFG data
        cfg_result = None
        try:
            node_dir = _node_helper_path()
            js_entry = os.path.join(node_dir, 'index.js')
            
            if os.path.exists(js_entry):
                proc = subprocess.Popen(
                    ['node', js_entry],
                    cwd=node_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                
                input_payload = json.dumps({"code": code})
                stdout, stderr = proc.communicate(input=input_payload, timeout=30)
                
                if proc.returncode == 0:
                    cfg_result = json.loads(stdout)
        except Exception as e:
            # Continue without CFG if parsing fails
            print(f"CFG generation failed: {e}")
        
        # Run security analysis
        analyzer = SecurityAnalyzer()
        analysis_result = analyzer.analyze(
            code=code,
            cfg_data=cfg_result
        )
        
        # Combine CFG with vulnerability info
        if cfg_result:
            analysis_result['cfg'] = cfg_result
            
            # Mark vulnerable nodes
            vuln_node_ids = {v['nodeId'] for v in analysis_result['vulnerabilities'] if v.get('nodeId')}
            
            for node in cfg_result.get('nodes', []):
                if node['id'] in vuln_node_ids:
                    # Find vulnerabilities for this node
                    node_vulns = [v for v in analysis_result['vulnerabilities'] if v.get('nodeId') == node['id']]
                    
                    # Get highest severity
                    severity_order = ['critical', 'high', 'medium', 'low', 'info']
                    max_severity = 'info'
                    for vuln in node_vulns:
                        vuln_sev = vuln['severity']
                        if severity_order.index(vuln_sev) < severity_order.index(max_severity):
                            max_severity = vuln_sev
                    
                    # Add vulnerability styling
                    node['data']['vulnerable'] = True
                    node['data']['severity'] = max_severity
                    node['data']['vulnerabilities'] = node_vulns
                    
                    # Update node style based on severity
                    if 'style' not in node:
                        node['style'] = {}
                    
                    severity_colors = {
                        'critical': '#d32f2f',
                        'high': '#f57c00',
                        'medium': '#fbc02d',
                        'low': '#7cb342',
                        'info': '#0288d1'
                    }
                    
                    node['style']['border'] = f"3px solid {severity_colors.get(max_severity, '#999')}"
                    node['style']['boxShadow'] = f"0 0 10px {severity_colors.get(max_severity, '#999')}"
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({"error": str(e), "type": "analysis_error"}), 500


