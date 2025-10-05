from flask import request, jsonify
from . import api_bp
import json
import os
import subprocess
import sys


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


