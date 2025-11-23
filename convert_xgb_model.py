import xgboost as xgb

# Load old model (binary .bst)
booster = xgb.Booster()
booster.load_model("models/xgb_model.bst")   # old format

# Save in NEW JSON format
booster.save_model("models/xgb_model.json")

print("Converted model saved as models/xgb_model.json")
