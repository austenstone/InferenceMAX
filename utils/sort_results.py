import sys
import json
from pathlib import Path

results = []
results_path = Path(sys.argv[1])

with open(results_path, 'r') as f:
    results = json.load(f)
    
results.sort(key=lambda r: (r['hw'], r.get('framework', 'vllm'), r.get('precision', 'fp8'), r['tp'], r['conc']))

output_path = results_path.parent / f"{results_path.stem}_sorted.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)