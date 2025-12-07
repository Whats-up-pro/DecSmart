import csv
import os
import time
import requests
import json

# Configuration
CSV_PATH = "contracts.csv"
OUTPUT_DIR = "data/solidity_bytecode"
RPC_URL = "https://eth.llamarpc.com" # Public RPC node

def fetch_bytecode(address):
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getCode",
        "params": [address, "latest"],
        "id": 1
    }
    try:
        response = requests.post(RPC_URL, json=payload, timeout=10)
        data = response.json()
        if 'result' in data:
            return data['result']
    except Exception as e:
        print(f"Error fetching {address}: {e}")
    return None

def main():
    if not os.path.exists(CSV_PATH):
        print(f"File {CSV_PATH} not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading {CSV_PATH}...")
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            address = row.get('address')
            if not address:
                continue
                
            # Check if already downloaded
            file_path = os.path.join(OUTPUT_DIR, f"{address}.bin")
            if os.path.exists(file_path):
                print(f"Skipping {address} (already exists)")
                continue
                
            print(f"Fetching bytecode for {address}...")
            bytecode = fetch_bytecode(address)
            
            if bytecode and bytecode != '0x':
                with open(file_path, 'w') as out_f:
                    out_f.write(bytecode)
                count += 1
            else:
                print(f"No bytecode found for {address}")
            
            # Rate limiting to be nice to public RPC
            time.sleep(0.2)
            
            if count >= 1000: # Limit to 1000 for demo purposes
                print("Downloaded 1000 contracts. Stopping for now.")
                break

    print(f"Done. Downloaded {count} contracts to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
