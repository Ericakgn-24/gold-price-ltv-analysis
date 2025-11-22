
import joblib
import xgboost as xgb
import torch
from lstm_model_def import LSTMModel

def load_rf():
    return joblib.load("models/random_forest_model.pkl")

def load_xgb():
    booster = xgb.Booster()
    booster.load_model("models/xgb_model.bst")
    return booster

def load_lstm(input_dim):
    model = LSTMModel(input_dim=input_dim)
    model.load_state_dict(torch.load("models/lstm_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model
