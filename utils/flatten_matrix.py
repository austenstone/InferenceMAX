import yaml
import os
import json

with open('.github/configs/search-space.yml', 'r') as f:
    search_space = yaml.safe_load(f)

seq_len_map = {
    '1024': '1k',
    '8192': '8k',
}

runner = os.environ['RUNNER']
model = os.environ['MODEL']
isl = os.environ['ISL']
osl = os.environ['OSL']
precision = os.environ['PRECISION']

# In the workflows, not all model references are the same so do a sort
# of partial matching to map to the search space keys
model_map = {
    'gpt-oss': 'gptoss',
    'llama-3.3-70b-instruct': 'llama',
    'deepseek-r1-0528': 'dsr1',
}

model_key = None
model_lower = model.lower()
for key, value in model_map.items():
    if key.lower() in model_lower:
        model_key = value
        break

if model_key is None:
    raise ValueError(f"Model '{model}' is not recognized.")

seq_len = f"{seq_len_map[isl]}{seq_len_map[osl]}"

if isinstance(search_space, list):
    entries = search_space
else:
    entries = search_space.get(model_key, {}).get(
        precision, {}).get(runner, {}).get(seq_len, [])

flattened_search_space = []
for entry in entries:
    tp_list = entry['tp'] if isinstance(
        entry.get('tp'), list) else [entry.get('tp')]

    conc_config = entry.get('conc')
    if isinstance(conc_config, dict):
        start = conc_config['start']
        end = conc_config['end']
        step_factor = conc_config.get('step', 2)
        conc_list = []
        current = start
        while current <= end:
            conc_list.append(current)
            current *= step_factor
    elif isinstance(conc_config, list):
        conc_list = conc_config
    else:
        conc_list = [conc_config]

    for tp_value in tp_list:
        for conc_value in conc_list:
            new_entry = entry.copy()
            new_entry['tp'] = tp_value
            new_entry['conc'] = conc_value
            flattened_search_space.append(new_entry)

with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
    f.write(f"flattened-matrix={json.dumps(flattened_search_space)}\n")
