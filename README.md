# MLOps Plan: KROM Bank Indonesia Stock Prediction

**Status**: In Development 🚀  
**Date Updated**: February 5, 2026  
**Python Version**: 3.13 (via venv)

---

## 📋 Project Overview

Rencana Machine Learning Operations (MLOps) lengkap untuk prediksi harga saham KROM Bank Indonesia (BBSI.JK) menggunakan data historis dari Kaggle. Proyek ini mencakup data ingestion, preprocessing, feature engineering, model training (RandomForest, Prophet, LSTM, CNN), experiment tracking (MLflow), dan persiapan deployment.

---

## ✅ Completed Tasks

### 1. Dataset Management
- ✅ Folder `dataset/` dibuat dengan struktur lengkap
- ✅ Dataset BBSI.JK berhasil diunduh via Kaggle CLI
- ✅ File CSV tersimpan:
  - `BBSI.JK.csv` (data harian)
  - `BBSI.JK_monthly.csv` (data bulanan)
  - `BBSI.JK_weekly.csv` (data mingguan)
- ✅ Metadata & snapshot tersimpan di `dataset/README.txt`

### 2. Notebook MLOps (`mlops_plan_krom_bank.ipynb`)
Notebook interaktif lengkap dengan 12+ sel:

| Cell | Deskripsi | Status |
|------|-----------|--------|
| 1 | Markdown: Judul & Overview | ✅ |
| 2 | Setup: pip update + install dependencies | ✅ |
| 3 | Imports umum & pembuatan folder | ✅ |
| 4 | Data Loading: CSV detection & load | ✅ |
| 5 | EDA: missing values, describe, date range | ✅ |
| 6 | Preprocessing: date index, resample, fill | ✅ |
| 7 | Feature Engineering: returns, lags, SMA | ✅ |
| 8 | Train/Val/Test Split: time-series aware | ✅ |
| 9 | RandomForest Baseline: pipeline + save | ✅ |
| 10 | Prophet + LSTM: contoh & skeleton | ✅ |
| 11 | MLflow: tracking lokal & logging | ✅ |
| 12 | CNN (Conv1D): model + save artifacts | ✅ |
| 13 | Markdown: Next steps checklist | ✅ |

### 3. Environment & Security
- ✅ Virtual environment (venv) configured dengan Python 3.13
- ✅ Pip upgraded to 26.0.1 (latest)
- ✅ `README_KAGGLE.md` dengan instruksi setup credentials (manual)
- ✅ `kaggle.json` tersimpan di `~/.kaggle/` (production credentials)
- ✅ `.gitignore` melindungi credentials & artifacts dari version control
- ✅ Template files dihapus (cleanup completed; gunakan `README_KAGGLE.md` untuk setup)

### 4. Folder Structure
```
Project_MachineLearning-1/
├── mlops_plan_krom_bank.ipynb       # Main notebook
├── dataset/                          # Data folder
│   ├── BBSI.JK.csv
│   ├── BBSI.JK_monthly.csv
│   ├── BBSI.JK_weekly.csv
│   ├── raw.csv                      # Snapshot
│   └── README.txt
├── models/                           # Trained models
│   ├── rf_baseline.pkl
│   └── cnn_conv1d/
├── artifacts/                        # Model artifacts
│   ├── preprocessed.parquet
│   ├── feature_list.txt
│   ├── scaler.joblib
│   └── mlruns/                      # MLflow tracking
├── .venv/                           # Virtual environment
├── .gitignore
├── README.md                        # This file
└── README_KAGGLE.md
```

---

## ⏳ In Progress / TODO

### Immediate (Next Steps)
- [ ] Run notebook cell 1-2 (setup & pip install)
- [ ] Run notebook cells 3-5 (data load & EDA)
- [ ] Run notebook cells 6-7 (preprocessing & feature engineering)
- [ ] Run notebook cells 8-9 (baseline & RandomForest training)
- [ ] Run notebook cell 12 (CNN training & save artifacts)
- [ ] Verify outputs in `models/` and `artifacts/`

### Medium Term
- [ ] Install & run Prophet baseline model
- [ ] Train LSTM model end-to-end
- [ ] Validate preprocessing artifacts (parquet, scaler)
- [ ] Log all experiments to MLflow
- [ ] Compare model metrics (RMSE, MAE, MAPE)

### Long Term (Production Readiness)
- [ ] Add unit tests (`pytest`) untuk preprocessing & feature engineering
- [ ] Add data validation dengan `pandera`
- [ ] Build Dockerfile untuk containerization
- [ ] Create FastAPI endpoint (`/predict`)
- [ ] Setup CI/CD (GitHub Actions) untuk lint, test, build image
- [ ] Add monitoring: drift detection & retraining triggers
- [ ] Deploy ke staging/production environment

---

## 📊 Models & Approaches

### Baseline Models
1. **Naive Forecast**: predict(t+1) = close(t)
2. **Moving Average**: simple MA-based forecast

### Statistical Models
- **ARIMA/SARIMAX**: (implemented as optional)
- **Prophet**: Facebook's time-series forecasting library

### Machine Learning
- **RandomForest**: ensemble regressor dengan pipeline preprocessing
- **LightGBM**: gradient boosting (optional, listed in dependencies)

### Deep Learning
- **LSTM**: Long Short-Term Memory neural network (1 layer, 32 units)
- **CNN (Conv1D)**: Convolutional neural network dengan 2 conv blocks + dense layers

---

## 🚀 Quick Start

### Prerequisites
1. Python 3.13 (installed)
2. Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Setup
```bash
# 1. Create venv with Python 3.13
py -3.13 -m venv .venv

# 2. Activate venv
.venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels \
    tensorflow mlflow optuna joblib pandera lightgbm

# 4. Download dataset (if not done)
kaggle datasets download -d caesarmario/krom-bank-indonesia-stock-historical-price -p dataset --unzip
```

### Run Notebook
```bash
# Open notebook in VS Code or Jupyter
jupyter notebook mlops_plan_krom_bank.ipynb
```

Then run cells sequentially (1 → 13):
- Cells 1-2: Setup & imports
- Cells 3-7: Data load, EDA, preprocessing, features
- Cells 8-9: Split & baseline models
- Cells 10-12: Deep learning models & MLflow
- Cell 13: Next steps

---

## 📈 Metrics & Evaluation

Models evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

Train/Val/Test split: **70% / 15% / 15%** (time-series aware)

---

## 📝 Notes

- **TensorFlow & Prophet**: Membutuhkan paket besar; instalasi mungkin memakan waktu
- **Dataset**: Diunduh dari Kaggle; pastikan kredensial valid di `~/.kaggle/kaggle.json`
- **MLflow**: Tracking lokal di `artifacts/mlruns/`; bisa dilihat via `mlflow ui`
- **Security**: Jangan commit `kaggle.json` atau credentials ke repository

---

## 📞 Contact & References

- **Dataset**: [KROM Bank Indonesia Stock Historical Price](https://www.kaggle.com/datasets/caesarmario/krom-bank-indonesia-stock-historical-price)
- **Libraries**: TensorFlow, scikit-learn, pandas, MLflow, Prophet
- **MLOps Best Practices**: Model versioning, experiment tracking, automated retraining

---

**Last Updated**: February 5, 2026