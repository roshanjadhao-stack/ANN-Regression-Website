# PowerPulse — ANN Regression Prediction Website

A Streamlit web app using the supplied PyTorch ANN model to estimate power plant energy output from AT, V, AP, and RH.

## Files
- `app.py` — web application
- `best_model.pt` — trained ANN weights
- `powerplant_data.csv` — dataset used by the notebook
- `ANN_Regression.ipynb` — original notebook
- `requirements.txt` — Python dependencies

## Run locally (Windows PowerShell)

```powershell
cd "path\to\ANN-Regression-Website"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Model details
- Input features: AT, V, AP, RH
- Target: PE
- Network: 4 → 6 (ReLU) → 6 (ReLU) → 1
- Evaluation: 80/20 split, `random_state=42`; StandardScaler fit on training features.

The app reconstructs the scaler using the same dataset split as the notebook because the original ZIP did not include a saved scaler. Keep the CSV unchanged for consistent predictions.

## Evaluation
The bundled model evaluated on the reconstructed held-out split at build time with approximately:
- R²: 0.9320
- MAE: 3.47 MW
- RMSE: 4.41 MW

Metrics can vary if the dataset, weights, preprocessing, or library versions change. Educational demo only; not for real-world plant control.
