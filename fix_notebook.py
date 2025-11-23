import json
import os

nb_path = "Notebooks/gold_price_ml.ipynb"

def fix_notebook():
    if not os.path.exists(nb_path):
        print(f"Notebook not found at {nb_path}")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Fix os.chdir
    # Look for cell with os.chdir
    chdir_fixed = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            new_source = []
            changed = False
            for line in source:
                if 'os.chdir("C:/Users/brahm/Desktop/GOLD-PRICE-LTV-ANALYSIS")' in line:
                    new_source.append('# ' + line.replace('\n', '') + ' # Hardcoded path removed\n')
                    new_source.append('# Ensure we are in the project root if running from Notebooks dir\n')
                    new_source.append('if os.getcwd().endswith("Notebooks"):\n')
                    new_source.append('    os.chdir("..")\n')
                    changed = True
                    chdir_fixed = True
                else:
                    new_source.append(line)
            if changed:
                cell["source"] = new_source
                print("Fixed os.chdir in notebook.")
                break
    
    if not chdir_fixed:
        print("os.chdir not found or already fixed.")

    # 2. Add feature saving after joblib.dump
    # Look for cell with joblib.dump(rf, ...)
    save_fixed = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            new_source = []
            changed = False
            for line in source:
                new_source.append(line)
                if 'joblib.dump(rf, "models/random_forest_model.pkl")' in line:
                    new_source.append('\n')
                    new_source.append('# Save feature names for app.py compatibility\n')
                    new_source.append('if hasattr(rf, "feature_names_in_"):\n')
                    new_source.append('    import json\n')
                    new_source.append('    with open("models/rf_feature_names.json", "w") as f:\n')
                    new_source.append('        json.dump(list(rf.feature_names_in_), f)\n')
                    new_source.append('    print("Saved RF feature names to models/rf_feature_names.json")\n')
                    changed = True
                    save_fixed = True
            if changed:
                cell["source"] = new_source
                print("Added feature saving code to notebook.")
                break

    if not save_fixed:
        print("joblib.dump not found or already fixed.")

    # Save back
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook saved to {nb_path}")

if __name__ == "__main__":
    fix_notebook()
