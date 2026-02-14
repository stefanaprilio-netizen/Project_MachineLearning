import json
import os

notebook_path = r'c:\Users\ASUS\Documents\MyProject_Machine\Project_MachineLearning\mlops_plan_krom_bank.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the first cell to include ipywidgets
packages_line_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "'lightgbm==4.6.0'" in line:
                source[i] = line.replace("'lightgbm==4.6.0'", "'lightgbm==4.6.0', 'ipywidgets'")
                packages_line_found = True
                break
        if packages_line_found:
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated notebook packages list.")
