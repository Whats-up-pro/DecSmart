"""
Security Vulnerability Analyzer for Solidity Smart Contracts
Analyzes AST and CFG to detect common vulnerabilities
"""
import re
from typing import List, Dict, Any, Set


class Vulnerability:
    """Represents a detected security vulnerability"""
    
    SEVERITY_CRITICAL = "critical"
    SEVERITY_HIGH = "high"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_LOW = "low"
    SEVERITY_INFO = "info"
    
    def __init__(self, vuln_type: str, severity: str, line: int, 
                 description: str, recommendation: str, node_id: str = None):
        self.type = vuln_type
        self.severity = severity
        self.line = line
        self.description = description
        self.recommendation = recommendation
        self.node_id = node_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "line": self.line,
            "description": self.description,
            "recommendation": self.recommendation,
            "nodeId": self.node_id
        }


class SecurityAnalyzer:
    """Main security analyzer class"""
    
    def __init__(self):
        self.vulnerabilities: List[Vulnerability] = []
        self.visited_nodes: Set[str] = set()
    
    def analyze(self, code: str, ast_data: Dict = None, cfg_data: Dict = None) -> Dict[str, Any]:
        """
        Analyze Solidity code for security vulnerabilities
        
        Args:
            code: Raw Solidity source code
            ast_data: Abstract Syntax Tree data (optional)
            cfg_data: Control Flow Graph data with nodes and edges
        
        Returns:
            Dictionary containing vulnerabilities and statistics
        """
        self.vulnerabilities = []
        self.visited_nodes = set()
        
        # Analyze code patterns
        self._check_reentrancy(code)
        self._check_unchecked_external_calls(code)
        self._check_integer_overflow(code)
        self._check_tx_origin(code)
        self._check_unprotected_selfdestruct(code)
        self._check_delegatecall(code)
        self._check_timestamp_dependence(code)
        self._check_uninitialized_storage(code)
        self._check_access_control(code)
        self._check_dos_vulnerabilities(code)
        
        # Analyze CFG if provided
        if cfg_data and 'nodes' in cfg_data:
            self._analyze_cfg(cfg_data, code)
        
        return {
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "summary": self._generate_summary(),
            "score": self._calculate_security_score()
        }
    
    def _analyze_cfg(self, cfg_data: Dict, code: str):
        """Analyze control flow graph for vulnerabilities"""
        nodes = cfg_data.get('nodes', [])
        edges = cfg_data.get('edges', [])
        
        # Check for unreachable code
        self._check_unreachable_code(nodes, edges)
        
        # Check for infinite loops
        self._check_infinite_loops(nodes, edges)
        
        # Map vulnerabilities to CFG nodes
        self._map_vulnerabilities_to_nodes(nodes, code)
    
    def _check_reentrancy(self, code: str):
        """Detect potential reentrancy vulnerabilities"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for external call followed by state change
            if re.search(r'\.(call|send|transfer)\s*\{', line, re.IGNORECASE):
                # Look ahead for state changes (skip require/assert lines)
                for j in range(i, min(i + 15, len(lines))):
                    line_content = lines[j].strip()
                    # Skip empty lines, comments, and control statements
                    if not line_content or line_content.startswith('//'):
                        continue
                    
                    # Check for state changes (but not in require/assert context)
                    if re.search(r'(balances\[.*?\]|storage\s+\w+)\s*(=|\+=|-=)', lines[j]):
                        # Check if this is not inside require/assert
                        if not re.search(r'(require|assert|revert)\s*\(', lines[j]):
                            self.vulnerabilities.append(Vulnerability(
                                vuln_type="Reentrancy",
                                severity=Vulnerability.SEVERITY_CRITICAL,
                                line=i,
                                description=f"Potential reentrancy vulnerability: External call on line {i} followed by state change on line {j+1}",
                                recommendation="Use checks-effects-interactions pattern: update state BEFORE external calls, or use ReentrancyGuard modifier"
                            ))
                            break
    
    def _check_unchecked_external_calls(self, code: str):
        """Detect unchecked return values from external calls"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for .call, .send without checking return value OR checking but not using require
            if re.search(r'\.(call|send)\s*\{', line):
                # Check if return value is captured
                if re.search(r'\(\s*bool\s+\w+\s*,', line):
                    # Return value is captured, check if it's validated in next few lines
                    found_require = False
                    for j in range(i, min(i + 3, len(lines))):
                        if re.search(r'require\s*\(\s*\w+\s*,', lines[j]):
                            found_require = True
                            break
                    
                    if not found_require:
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type="Unchecked Call Return Value",
                            severity=Vulnerability.SEVERITY_HIGH,
                            line=i,
                            description=f"Return value captured but not validated with require() on line {i}",
                            recommendation="Add require(success, 'Call failed') to validate the call succeeded"
                        ))
                else:
                    # Return value not captured at all
                    if not re.search(r'require\s*\(.*\.(call|send)', line):
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type="Unchecked Call Return Value",
                            severity=Vulnerability.SEVERITY_HIGH,
                            line=i,
                            description=f"Unchecked return value from external call on line {i}",
                            recommendation="Always check return values: (bool success, ) = address.call(...); require(success, 'Call failed');"
                        ))
    
    def _check_integer_overflow(self, code: str):
        """Detect potential integer overflow/underflow"""
        lines = code.split('\n')
        
        # Check Solidity version
        has_safe_math = re.search(r'pragma\s+solidity\s+\^?0\.[0-7]\.', code)
        
        if has_safe_math:
            for i, line in enumerate(lines, 1):
                if re.search(r'(\+\+|--|\+=|-=|\*=|/=)', line):
                    if not re.search(r'SafeMath|unchecked', code):
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type="Integer Overflow/Underflow",
                            severity=Vulnerability.SEVERITY_HIGH,
                            line=i,
                            description=f"Potential integer overflow/underflow on line {i} (Solidity < 0.8.0)",
                            recommendation="Use SafeMath library or upgrade to Solidity ^0.8.0"
                        ))
                        break  # Report once per contract
    
    def _check_tx_origin(self, code: str):
        """Detect usage of tx.origin for authorization"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if re.search(r'tx\.origin', line, re.IGNORECASE):
                self.vulnerabilities.append(Vulnerability(
                    vuln_type="tx.origin Authentication",
                    severity=Vulnerability.SEVERITY_MEDIUM,
                    line=i,
                    description=f"Using tx.origin for authentication on line {i}",
                    recommendation="Use msg.sender instead of tx.origin for authorization checks"
                ))
    
    def _check_unprotected_selfdestruct(self, code: str):
        """Detect unprotected selfdestruct calls"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if re.search(r'selfdestruct\s*\(', line, re.IGNORECASE):
                # Check if there's access control nearby
                context = '\n'.join(lines[max(0, i-5):min(len(lines), i+2)])
                if not re.search(r'(require|modifier|onlyOwner|onlyAdmin)', context, re.IGNORECASE):
                    self.vulnerabilities.append(Vulnerability(
                        vuln_type="Unprotected Selfdestruct",
                        severity=Vulnerability.SEVERITY_CRITICAL,
                        line=i,
                        description=f"Unprotected selfdestruct call on line {i}",
                        recommendation="Add access control modifier (e.g., onlyOwner) to selfdestruct function"
                    ))
    
    def _check_delegatecall(self, code: str):
        """Detect dangerous delegatecall usage"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if re.search(r'\.delegatecall\s*\(', line, re.IGNORECASE):
                self.vulnerabilities.append(Vulnerability(
                    vuln_type="Dangerous Delegatecall",
                    severity=Vulnerability.SEVERITY_HIGH,
                    line=i,
                    description=f"Delegatecall usage on line {i} - can be dangerous if target is user-controlled",
                    recommendation="Ensure delegatecall target is trusted and validated. Consider using a whitelist."
                ))
    
    def _check_timestamp_dependence(self, code: str):
        """Detect timestamp dependence"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if re.search(r'(block\.timestamp|now)\s*(==|<|>|<=|>=)', line, re.IGNORECASE):
                self.vulnerabilities.append(Vulnerability(
                    vuln_type="Timestamp Dependence",
                    severity=Vulnerability.SEVERITY_MEDIUM,
                    line=i,
                    description=f"Timestamp dependence on line {i} - miners can manipulate timestamps",
                    recommendation="Avoid using block.timestamp for critical logic. Use block.number or oracle services."
                ))
    
    def _check_uninitialized_storage(self, code: str):
        """Detect uninitialized storage pointers"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Look for storage variables without initialization
            if re.search(r'\s+storage\s+\w+\s*;', line):
                self.vulnerabilities.append(Vulnerability(
                    vuln_type="Uninitialized Storage Pointer",
                    severity=Vulnerability.SEVERITY_HIGH,
                    line=i,
                    description=f"Uninitialized storage pointer on line {i}",
                    recommendation="Always initialize storage pointers or use memory keyword"
                ))
    
    def _check_access_control(self, code: str):
        """Detect missing access control"""
        lines = code.split('\n')
        
        # Look for state-changing functions without modifiers
        in_function = False
        function_line = 0
        function_visibility = ""
        
        for i, line in enumerate(lines, 1):
            func_match = re.search(r'function\s+(\w+)\s*\([^)]*\)\s*(public|external)?\s*(.*?)\s*\{?', line, re.IGNORECASE)
            if func_match:
                in_function = True
                function_line = i
                function_visibility = func_match.group(2) or "public"
                modifiers = func_match.group(3) or ""
                
                # Check for state-changing functions without access control
                if function_visibility in ['public', 'external']:
                    if not re.search(r'(only\w+|require\s*\(\s*msg\.sender)', modifiers, re.IGNORECASE):
                        # Check if function changes state
                        func_body_start = i
                        func_body_end = min(i + 20, len(lines))
                        func_body = '\n'.join(lines[func_body_start:func_body_end])
                        
                        if re.search(r'(=|\+=|-=|\.transfer|\.send|\.call|selfdestruct)', func_body):
                            self.vulnerabilities.append(Vulnerability(
                                vuln_type="Missing Access Control",
                                severity=Vulnerability.SEVERITY_HIGH,
                                line=function_line,
                                description=f"Public/external state-changing function without access control on line {function_line}",
                                recommendation="Add access control modifier or require statement to restrict access"
                            ))
    
    def _check_dos_vulnerabilities(self, code: str):
        """Detect potential Denial of Service vulnerabilities"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for loops with external calls
            if re.search(r'for\s*\(', line, re.IGNORECASE):
                # Look ahead for external calls in loop body
                for j in range(i, min(i + 15, len(lines))):
                    if re.search(r'\.(call|send|transfer)\s*\(', lines[j]):
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type="DoS with Block Gas Limit",
                            severity=Vulnerability.SEVERITY_MEDIUM,
                            line=i,
                            description=f"Loop starting at line {i} contains external call - potential gas limit DoS",
                            recommendation="Avoid loops with external calls. Use pull over push pattern."
                        ))
                        break
    
    def _check_unreachable_code(self, nodes: List[Dict], edges: List[Dict]):
        """Detect unreachable nodes in CFG"""
        # Build adjacency list
        incoming = {node['id']: [] for node in nodes}
        for edge in edges:
            if edge['target'] in incoming:
                incoming[edge['target']].append(edge['source'])
        
        # Find entry nodes
        entry_nodes = [node for node in nodes if node.get('type') == 'entry']
        
        if not entry_nodes:
            return
        
        # BFS to find reachable nodes
        reachable = set()
        queue = [node['id'] for node in entry_nodes]
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            for edge in edges:
                if edge['source'] == current and edge['target'] not in reachable:
                    queue.append(edge['target'])
        
        # Report unreachable nodes (but skip function labels and normal nodes)
        for node in nodes:
            if node['id'] not in reachable and node.get('type') not in ['entry', 'exit', 'default']:
                line = node.get('data', {}).get('startLine', 0)
                # Only report if it's actual code, not metadata
                if line and line > 1:  # Skip line 1 which is usually pragma
                    self.vulnerabilities.append(Vulnerability(
                        vuln_type="Unreachable Code",
                        severity=Vulnerability.SEVERITY_INFO,
                        line=line,
                        description=f"Unreachable code detected at line {line}",
                        recommendation="Remove dead code or fix control flow logic",
                        node_id=node['id']
                    ))
    
    def _check_infinite_loops(self, nodes: List[Dict], edges: List[Dict]):
        """Detect potential infinite loops"""
        # Build adjacency list
        graph = {}
        for node in nodes:
            graph[node['id']] = []
        
        for edge in edges:
            if edge['source'] in graph:
                graph[edge['source']].append(edge['target'])
        
        # Detect cycles without exit conditions
        def has_cycle(node_id, visited, rec_stack):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        visited = set()
        for node in nodes:
            if node['id'] not in visited:
                rec_stack = set()
                if has_cycle(node['id'], visited, rec_stack):
                    # Check if it's a loop construct
                    node_label = node.get('data', {}).get('label', '').lower()
                    if 'while' in node_label or 'for' in node_label:
                        line = node.get('data', {}).get('startLine', 0)
                        self.vulnerabilities.append(Vulnerability(
                            vuln_type="Potential Infinite Loop",
                            severity=Vulnerability.SEVERITY_MEDIUM,
                            line=line,
                            description=f"Potential infinite loop detected at line {line}",
                            recommendation="Ensure loop has proper exit condition and gas limits",
                            node_id=node['id']
                        ))
    
    def _map_vulnerabilities_to_nodes(self, nodes: List[Dict], code: str):
        """Map detected vulnerabilities to CFG nodes"""
        for vuln in self.vulnerabilities:
            if vuln.node_id:
                continue
            
            # Find node that corresponds to vulnerability line
            for node in nodes:
                start_line = node.get('data', {}).get('startLine', 0)
                end_line = node.get('data', {}).get('endLine', 999999)
                
                if start_line <= vuln.line <= end_line:
                    vuln.node_id = node['id']
                    break
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate vulnerability summary statistics"""
        severity_counts = {
            Vulnerability.SEVERITY_CRITICAL: 0,
            Vulnerability.SEVERITY_HIGH: 0,
            Vulnerability.SEVERITY_MEDIUM: 0,
            Vulnerability.SEVERITY_LOW: 0,
            Vulnerability.SEVERITY_INFO: 0,
        }
        
        type_counts = {}
        
        for vuln in self.vulnerabilities:
            severity_counts[vuln.severity] += 1
            type_counts[vuln.type] = type_counts.get(vuln.type, 0) + 1
        
        return {
            "total": len(self.vulnerabilities),
            "bySeverity": severity_counts,
            "byType": type_counts
        }
    
    def _calculate_security_score(self) -> int:
        """Calculate security score (0-100, higher is better)"""
        if not self.vulnerabilities:
            return 100
        
        # Weighted scoring
        weights = {
            Vulnerability.SEVERITY_CRITICAL: 20,
            Vulnerability.SEVERITY_HIGH: 10,
            Vulnerability.SEVERITY_MEDIUM: 5,
            Vulnerability.SEVERITY_LOW: 2,
            Vulnerability.SEVERITY_INFO: 1,
        }
        
        deductions = sum(weights[v.severity] for v in self.vulnerabilities)
        score = max(0, 100 - deductions)
        
        return score
