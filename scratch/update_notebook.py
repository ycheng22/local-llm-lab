import json
import os

p = r'notebooks/sft_evaluation.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    source = cell.get('source', [])
    if any('sft_v1_5k.yaml' in line and '!python' in line for line in source):
        cell['source'] = [
            'import sys\n',
            'import os\n',
            'sys.path.append(os.path.abspath("../src"))\n',
            'sys.argv = ["sft.py", "--config", "../configs/sft_v1_5k.yaml"]\n',
            'from llm_lab.training.sft import main\n',
            'main()\n'
        ]

with open(p, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
