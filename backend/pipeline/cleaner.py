import json
import os
import pandas as pd

class DataCleaner:
    def __init__(self, duplicates_path=None):
        self.duplicates = set()
        if duplicates_path and os.path.exists(duplicates_path):
            self.load_duplicates(duplicates_path)

    def load_duplicates(self, path):
        """
        Load duplicates from a JSON file (SmartBugs-wild format).
        Expected format: list of lists of duplicates, or a dict mapping.
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Assuming data is a list of groups of duplicates
                # We keep the first one, mark others as duplicates
                for group in data:
                    if isinstance(group, list) and len(group) > 1:
                        # Add all except the first one to duplicates set
                        for dup in group[1:]:
                            self.duplicates.add(dup)
            print(f"Loaded {len(self.duplicates)} duplicate entries to ignore.")
        except Exception as e:
            print(f"Error loading duplicates: {e}")

    def filter_contracts(self, csv_path):
        """
        Read contracts.csv and return a filtered DataFrame.
        """
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return []

        df = pd.read_csv(csv_path)
        
        # Filter by duplicates if 'address' or 'name' is in duplicates
        if self.duplicates:
            initial_len = len(df)
            # Assuming 'address' is the key
            if 'address' in df.columns:
                df = df[~df['address'].isin(self.duplicates)]
            print(f"Filtered duplicates: {initial_len} -> {len(df)}")

        return df.to_dict('records')
