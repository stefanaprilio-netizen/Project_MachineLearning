import json
import os

notebook_path = 'mlops_plan_krom_bank.ipynb'

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

modified = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. RandomForest
        if 'RandomForestRegressor' in source and 'joblib.dump' in source:
            if 'plt.show()' not in source:
                print("Adding RF plots...")
                cell['source'].append("\n")
                cell['source'].append("# Visualisasi RF\n")
                cell['source'].append("plt.figure(figsize=(10, 5))\n")
                cell['source'].append("plt.plot(y_test.values, label='Actual', color='blue')\n")
                cell['source'].append("plt.plot(preds, label='Predicted', color='red', linestyle='--')\n")
                cell['source'].append("plt.title('Random Forest: Actual vs Predicted')\n")
                cell['source'].append("plt.legend()\n")
                cell['source'].append("plt.show()\n")
                modified = True
        
        # 2. Prophet & LSTM
        if 'Prophet' in source and 'LSTM' in source and 'Prophet RMSE' in source:
            if 'plt.show()' not in source:
                print("Adding Prophet & LSTM plots...")
                # Add plot for Prophet
                for i, line in enumerate(cell['source']):
                    if "print('Prophet RMSE:'," in line:
                        cell['source'].insert(i+1, "            plt.figure(figsize=(10, 5))\n")
                        cell['source'].insert(i+2, "            plt.plot(p_test['y'].values, label='Actual')\n")
                        cell['source'].insert(i+3, "            plt.plot(fc_pred.values, label='Predicted')\n")
                        cell['source'].insert(i+4, "            plt.title('Prophet: Actual vs Predicted')\n")
                        cell['source'].insert(i+5, "            plt.legend()\n")
                        cell['source'].insert(i+6, "            plt.show()\n")
                        break
                
                # Add plot for LSTM
                if 'history = model.fit' in source:
                    cell['source'].append("\n")
                    cell['source'].append("# Visualisasi LSTM\n")
                    cell['source'].append("plt.figure(figsize=(10, 4))\n")
                    cell['source'].append("plt.plot(history.history['loss'], label='Train Loss')\n")
                    cell['source'].append("plt.plot(history.history['val_loss'], label='Val Loss')\n")
                    cell['source'].append("plt.title('LSTM: Training vs Validation Loss')\n")
                    cell['source'].append("plt.legend()\n")
                    cell['source'].append("plt.show()\n")
                    cell['source'].append("\n")
                    cell['source'].append("y_pred_scaled = model.predict(Xte)\n")
                    cell['source'].append("y_pred = scaler_y.inverse_transform(y_pred_scaled)\n")
                    cell['source'].append("y_actual = scaler_y.inverse_transform(yte.reshape(-1, 1))\n")
                    cell['source'].append("plt.figure(figsize=(10, 5))\n")
                    cell['source'].append("plt.plot(y_actual, label='Actual', color='blue')\n")
                    cell['source'].append("plt.plot(y_pred, label='Predicted', color='green', linestyle='--')\n")
                    cell['source'].append("plt.title('LSTM: Actual vs Predicted')\n")
                    cell['source'].append("plt.legend()\n")
                    cell['source'].append("plt.show()\n")
                modified = True

        # 3. CNN
        if 'models.Sequential' in source and 'Conv1D' in source and 'model.fit' in source:
            if 'plt.show()' not in source:
                print("Adding CNN plots...")
                for i, line in enumerate(cell['source']):
                    if 'model.fit(Xtr, ytr,' in line and 'history_cnn' not in line:
                        cell['source'][i] = line.replace('model.fit(Xtr, ytr,', 'history_cnn = model.fit(Xtr, ytr,')
                
                cell['source'].append("\n")
                cell['source'].append("# Visualisasi CNN\n")
                cell['source'].append("plt.figure(figsize=(10, 4))\n")
                cell['source'].append("plt.plot(history_cnn.history['loss'], label='Train Loss')\n")
                cell['source'].append("plt.plot(history_cnn.history['val_loss'], label='Val Loss')\n")
                cell['source'].append("plt.title('CNN: Training vs Validation Loss')\n")
                cell['source'].append("plt.legend()\n")
                cell['source'].append("plt.show()\n")
                cell['source'].append("\n")
                cell['source'].append("y_pred_scaled = model.predict(Xte)\n")
                cell['source'].append("y_pred = scaler_y.inverse_transform(y_pred_scaled)\n")
                cell['source'].append("y_actual = scaler_y.inverse_transform(yte.reshape(-1, 1))\n")
                cell['source'].append("plt.figure(figsize=(10, 5))\n")
                cell['source'].append("plt.plot(y_actual, label='Actual', color='blue')\n")
                cell['source'].append("plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')\n")
                cell['source'].append("plt.title('CNN: Actual vs Predicted')\n")
                cell['source'].append("plt.legend()\n")
                cell['source'].append("plt.show()\n")
                modified = True

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4, ensure_ascii=False)
    print('Success: Updated notebook with validation plots.')
else:
    print('No changes needed. Plots already exist or targets not found.')
