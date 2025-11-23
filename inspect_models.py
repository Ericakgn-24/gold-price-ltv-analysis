import joblib
import json
import xgboost as xgb
import torch
import torch.nn as nn
import os
import sys

# Define LSTM Model class (must match training)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def inspect_rf():
    try:
        rf = joblib.load("models/random_forest_model.pkl")
        print(f"RF Model loaded. N_features: {rf.n_features_in_}")
        if hasattr(rf, "feature_names_in_"):
            print(f"RF Feature Names ({len(rf.feature_names_in_)}): {rf.feature_names_in_}")
            with open("rf_feature_names.json", "w") as f:
                json.dump(list(rf.feature_names_in_), f)
            print("Saved RF feature names to rf_feature_names.json")
        else:
            print("RF model does not have feature_names_in_")
    except Exception as e:
        print(f"Error inspecting RF: {e}")

def inspect_xgb():
    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model("models/xgb_model.json")
        print(f"XGB Model loaded. Features: {len(xgb_model.feature_names)}")
        # print(f"XGB Feature Names: {xgb_model.feature_names}")
    except Exception as e:
        print(f"Error inspecting XGB: {e}")

def inspect_lstm():
    try:
        # Load the checkpoint
        checkpoint = torch.load("models/lstm_model.pt", map_location=torch.device('cpu'))
        
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict):
            print("LSTM Checkpoint is a dictionary/state_dict.")
            # Try to infer input dimension from weight shapes
            # lstm.weight_ih_l0 shape is (4*hidden_dim, input_dim)
            if 'lstm.weight_ih_l0' in checkpoint:
                weight_shape = checkpoint['lstm.weight_ih_l0'].shape
                print(f"LSTM weight_ih_l0 shape: {weight_shape}")
                # hidden_dim is weight_shape[0] / 4
                input_dim = weight_shape[1]
                print(f"Inferred LSTM Input Dimension: {input_dim}")
            elif 'model_state_dict' in checkpoint:
                 state_dict = checkpoint['model_state_dict']
                 if 'lstm.weight_ih_l0' in state_dict:
                    weight_shape = state_dict['lstm.weight_ih_l0'].shape
                    print(f"LSTM weight_ih_l0 shape: {weight_shape}")
                    input_dim = weight_shape[1]
                    print(f"Inferred LSTM Input Dimension: {input_dim}")
        elif isinstance(checkpoint, nn.Module):
             print("LSTM Checkpoint is a full model.")
             # Try to access input_dim if stored, or check layer weights
             # Assuming standard LSTM layer
             for name, param in checkpoint.named_parameters():
                 if 'weight_ih_l0' in name:
                     print(f"LSTM {name} shape: {param.shape}")
                     print(f"Inferred LSTM Input Dimension: {param.shape[1]}")
                     break
        else:
            print(f"Unknown LSTM checkpoint type: {type(checkpoint)}")

    except Exception as e:
        print(f"Error inspecting LSTM: {e}")

if __name__ == "__main__":
    print("Inspecting Models...")
    inspect_rf()
    inspect_xgb()
    inspect_lstm()
