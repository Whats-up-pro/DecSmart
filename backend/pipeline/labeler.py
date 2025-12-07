import os
import json
import glob

class SmartBugsLabeler:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.labels = {} # filename -> 1 (vuln) or 0 (clean)

    def generate_labels(self, output_path="data/labels.json"):
        """
        Walk through SmartBugs results directory and aggregate findings.
        Structure usually: tool_name/contract_address/result.json
        """
        if not os.path.exists(self.results_dir):
            print(f"Results directory {self.results_dir} does not exist.")
            return

        print(f"Scanning {self.results_dir} for vulnerability reports...")
        
        # This is a simplified parser for SmartBugs results
        # We assume if any tool reports a vulnerability, it's vulnerable (Union approach)
        
        # Walk through all result.json files
        for result_file in glob.glob(os.path.join(self.results_dir, "**", "result.json"), recursive=True):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract contract name/address from path or json
                # Path example: results/mythril/0x123.../result.json
                path_parts = result_file.split(os.sep)
                contract_id = path_parts[-2] # 0x123...
                
                # Check for vulnerabilities
                is_vulnerable = self._check_vulnerability(data)
                
                # Update label: If already marked vulnerable, keep it. If not, update.
                current_label = self.labels.get(contract_id, 0)
                if is_vulnerable:
                    self.labels[contract_id] = 1
                    
            except Exception as e:
                # print(f"Error reading {result_file}: {e}")
                pass

        # Save labels
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.labels, f, indent=2)
            
        print(f"Generated labels for {len(self.labels)} contracts. Saved to {output_path}")
        return self.labels

    def _check_vulnerability(self, data):
        """
        Heuristic to check if a result JSON indicates a vulnerability.
        Adapts to different tools (Mythril, Slither, Oyente).
        """
        # Mythril
        if 'issues' in data and isinstance(data['issues'], list):
            if len(data['issues']) > 0:
                return True
                
        # Slither
        if 'analysis' in data:
            # Slither often puts results in 'analysis' list
            if isinstance(data['analysis'], list) and len(data['analysis']) > 0:
                return True
                
        # Oyente
        if 'vulnerabilities' in data:
             # Oyente format varies, sometimes it has 'callstack', 'reentrancy' keys
             for key, value in data['vulnerabilities'].items():
                 if value is True or (isinstance(value, list) and len(value) > 0):
                     return True

        return False
