# Rencana MLOps: Prediksi Saham KROM Bank Indonesia

**Status**: Dalam Pengembangan 🚀  
**Tanggal Diperbarui**: 14 Februari 2026  
**Versi Python**: 3.13 (via venv)

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
| ------ | ----------- | -------- |
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

**Recent updates in notebook:**

- ✔ ARIMA forecasting cell now uses rolling predictions and optional `auto_arima` for better accuracy; fixed numpy indexing bug.
- ✔ LSTM and CNN training steps expanded with deeper architectures, callbacks (ReduceLROnPlateau, EarlyStopping), longer epochs; training loss now can approach zero.
- ✔ Added Kaggle download snippet and environment configuration notes; dataset successfully downloaded into `dataset/`.
- ✔ Data validation with Pandera and unit test scaffolding added in later cells.

### 3. Environment & Security

- ✅ Virtual environment (venv) configured dengan Python 3.13
- ✅ Pip upgraded to 26.0.1 (latest)
- ✅ `README_KAGGLE.md` dengan instruksi setup credentials (manual)
- ✅ `kaggle.json` template tersedia di root (diabaikan oleh git)
- ✅ `requirements.txt` dibuat dengan versi library terbaru (Feb 2026)
- ✅ `kaggle.json` production tersimpan di `~/.kaggle/` (cek template)
- ✅ `.gitignore` melindungi credentials & artifacts dari version control
- ✅ Template files dihapus (cleanup completed; gunakan `README_KAGGLE.md` untuk setup)

### 4. Folder Structure

```text
Project_MachineLearning/
├── mlops_plan_krom_bank.ipynb       # Main notebook
├── main.py                           # FastAPI serving endpoint
├── Dockerfile                        # Containerization schema
├── test_preprocessing.py             # Unit tests for preprocessing
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI/CD GitHub Actions
├── dataset/                          # Data folder
│   ├── BBSI.JK.csv
│   ├── BBSI.JK_monthly.csv
│   ├── BBSI.JK_weekly.csv
│   ├── raw.csv                      # Snapshot
│   └── README.txt
├── models/                           # Trained models
│   ├── rf_baseline.pkl
│   └── cnn_conv1d.keras             # CNN Saved Model
├── artifacts/                        # Model artifacts
│   ├── preprocessed.parquet
│   ├── feature_list.txt
│   ├── scaler_x.joblib              # Feature scaler
│   ├── scaler_y.joblib              # Target scaler
│   └── arima_summary.txt            # ARIMA stats summary
├── .venv/                           # Virtual environment
├── .gitignore
├── requirements.txt                 # Pin dependencies
├── README.md                        # This file
└── README_KAGGLE.md
```

---

## ⏳ In Progress / TODO

### Immediate (Next Steps)

- ✅ Run notebook cell 1-2 (setup & pip install)
- ✅ Run notebook cells 3-5 (data load & EDA)
- ✅ Run notebook cells 6-7 (preprocessing & feature engineering)
- ✅ Run notebook cells 8-9 (baseline & RandomForest training)
- ✅ Run notebook cell 12 (CNN training & save artifacts)
- ✅ Verify outputs in `models/` and `artifacts/`

### Medium Term

- ✅ Install & run Prophet baseline model
- ✅ Train LSTM model end-to-end
- ✅ Validate preprocessing artifacts (parquet, scaler)
- ✅ Log all experiments to MLflow
- ✅ Compare model metrics (RMSE, MAE, MAPE)

### Long Term (Production Readiness)

- ✅ Add unit tests (`pytest`) untuk preprocessing & feature engineering
- ✅ Add data validation dengan `pandera`
- ✅ Build Dockerfile untuk containerization
- ✅ Create FastAPI endpoint (`/predict`)
- ✅ Setup CI/CD (GitHub Actions) untuk lint, test, build image
- ✅ Add monitoring: drift detection & retraining triggers
- ✅ Deploy ke staging/production environment

---

## 📊 Model & Pendekatan

### Model Baseline

1. **Prediksi Naif**: prediksi(t+1) = close(t)
2. **Moving Average**: prediksi berbasis MA sederhana

### Model Statistik

- **ARIMA/SARIMAX**: (diimplementasikan sebagai opsional)
- **Prophet**: Pustaka time-series forecasting dari Facebook

### Machine Learning

- **RandomForest**: ensemble regressor dengan preprocessing pipeline
- **LightGBM**: gradient boosting (opsional, tersedia di dependencies)

### Deep Learning

- **LSTM**: Long Short-Term Memory neural network (1 layer, 32 units)
- **CNN (Conv1D)**: Convolutional neural network dengan 2 conv blocks + dense layers

---

## 🚀 Quick Start

### Prasyarat

1. **Python 3.13 (wajib)** — TensorFlow & MLflow kompatibel hingga 3.13 saja. Python 3.14+ tidak didukung.
2. Kredensial API Kaggle (`~/.kaggle/kaggle.json`)

### Setup

```bash
# 1. Buat & aktifkan venv
py -3.13 -m venv .venv
.venv\Scripts\activate

# 2. Install semua dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Unduh dataset (sekali saja)
kaggle datasets download -d caesarmario/krom-bank-indonesia-stock-historical-price -p dataset --unzip
```

**⏱️ Catatan**: Instalasi pertama kali mungkin memakan waktu 5-10 menit (TensorFlow ~500MB) tergantung pada kecepatan koneksi internet Anda. Run berikutnya akan jauh lebih cepat. Jika koneksi tidak stabil, lihat bagian **Troubleshooting** di bawah.

### Alternatif: Menggunakan Anaconda

```bash
# 1. Buat conda environment dengan Python 3.13
conda create -n mlops-krom python=3.13

# 2. Aktifkan conda environment
conda activate mlops-krom

# 3. Install dependencies via conda (lebih cepat untuk TensorFlow)
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn statsmodels mlflow optuna joblib pandera lightgbm
conda install tensorflow

# 4. Install Kaggle via pip
pip install kaggle

# 5. Unduh dataset (sekali saja)
kaggle datasets download -d caesarmario/krom-bank-indonesia-stock-historical-price -p dataset --unzip
```

**✨ Tips**: Conda menangani package besar (seperti TensorFlow) lebih handal. Gunakan ini jika Anaconda/Miniconda sudah terinstall.

### Alternatif: Menggunakan Google Colab

```python
# 1. Buka Google Colab dan jalankan di cell pertama
!pip install --upgrade pip
!pip install pandas==3.0.1 numpy==2.4.2 matplotlib==3.10.8 seaborn==0.13.2 scikit-learn==1.8.0 statsmodels==0.15.0 tensorflow==2.20.0 mlflow==3.9.0 prophet==1.3.0 optuna==4.7.0 joblib==1.5.3 pandera==0.29.0 lightgbm==4.6.0

# 2. Mount Google Drive (untuk akses file)
from google.colab import drive
drive.mount('/content/drive')

# 3. Upload kaggle.json ke Colab
# Buka: https://www.kaggle.com/settings/account
# Download kaggle.json, lalu upload ke Colab via file manager (⬅️ panel kiri)
# Atau jalankan:
from google.colab import files
files.upload()  # Upload kaggle.json

# 4. Setup Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 5. Unduh dataset
!kaggle datasets download -d caesarmario/krom-bank-indonesia-stock-historical-price -p dataset --unzip

# 6. Clone repo atau upload notebook
!git clone https://github.com/stefanaprilio-netizen/Project_MachineLearning.git
# atau upload mlops_plan_krom_bank.ipynb langsung ke Colab
```

**⚠️ Penting untuk Colab**:

- **Disk Space**: Colab memberikan ~100GB storage, cukup untuk dataset & model
- **Runtime**: Setiap session baru. Save outputs ke Google Drive dengan:

  ```python
  # Save model ke Drive
  !cp -r models /content/drive/MyDrive/mlops-backup/
  !cp -r artifacts /content/drive/MyDrive/mlops-backup/
  ```

- **GPU/TPU**: Colab gratis dengan GPU. Aktifkan di: Runtime → Change runtime type → GPU
- **Kaggle API**: Pastikan `~/.kaggle/kaggle.json` ada dengan permission 600
- **TensorFlow**: Sudah preinstalled di Colab (versi terbaru). Jalankan `!pip install --upgrade tensorflow` jika perlu update

**Tips Colab**:

- Gunakan `!` untuk shell commands, `%` untuk magic commands
- Mount Drive untuk persistent storage: `from google.colab import drive; drive.mount('/content/drive')`
- Download hasil: `files.download('model.pkl')` atau save ke Drive

### Alternatif: Menggunakan GitHub Desktop

GitHub Desktop mempermudah management repository tanpa perlu command line. Ikuti langkah berikut:

#### Setup Awal

1. Download & Install GitHub Desktop dari <https://desktop.github.com>
2. Login dengan akun GitHub Anda
3. Klik "File" → "Clone Repository"
4. Pilih: stefanaprilio-netizen/Project_MachineLearning
5. Pilih folder lokal (misal: C:\Users\YourName\Documents\GitHub\)
6. Klik "Clone"

#### Workflow Harian (Commit & Push Changes)

**Setelah mengedit file (notebook, README, etc):**

1. **Lihat Changes** (di panel kiri):
   - Semua file yang berubah akan muncul di tab "Changes"
   - Biru = modified, Hijau = added

2. **Pilih file untuk di-stage** (checkbox sebelah filename):

   ☑ mlops_plan_krom_bank.ipynb
   ☑ README.md
   ☐ .venv/  (jangan, sudah di .gitignore)

3. **Write Commit Message** (di bawah panel Changes):

   Summary: Update notebook with CNN model training
   Description: Added Conv1D layers, saved preprocessing artifacts

4. **Klik "Commit to main"** (atau branch lain)

5. **Push ke GitHub**:
   - Klik tombol "Push origin" di top
   - Tunggu hingga selesai (muncul "No local changes")

#### Workflow Kolaborasi (Update dari Remote)

**Sebelum mulai bekerja di pagi hari:**

1. Klik tombol "Fetch origin" atau "Pull origin"
   - Fetch: Cek update dari GitHub tanpa merge
   - Pull: Download & merge update ke local

2. Jika ada conflict:
   - GitHub Desktop akan notifikasi
   - Pilih "Open in External Editor" untuk resolve conflict
   - Edit file, hapus conflict markers (`<<<<`, `>>>>`)
   - Save file, commit, push

#### Melihat History & Revert Changes

**History (untuk lihat commit sebelumnya):**

- Klik tab "History" di panel kiri
- Klik commit untuk lihat file yang berubah
- Klik file untuk lihat diff (perubahan detail)

**Revert commit (batalkan last commit):**

- Klik commit di History
- Klik "Revert this commit"
- GitHub Desktop membuat commit baru yang batalkan perubahan

**Discard changes (hapus local edits):**

- Di tab "Changes", klik file
- Klik "Discard changes"
- ⚠️ Tidak bisa di-undo, pastikan sudah backup!

#### Tips GitHub Desktop

| Aksi | Via GitHub Desktop | Via Command Line |
| ----- | ------------------- | ------------------ |
| Lihat status | Tab "Changes" | `git status` |
| Commit | Write message + Click "Commit" | `git commit -m "msg"` |
| Push | Click "Push origin" | `git push` |
| Pull | Click "Pull origin" | `git pull` |
| Lihat history | Tab "History" | `git log --oneline` |
| Buat branch baru | Click "Branch" → "New Branch" | `git branch <name>` |
| Switch branch | Click branch di tab "Current Branch" | `git checkout <name>` |
| Stash changes | Klik branch, pilih stash | `git stash` |

#### Troubleshooting GitHub Desktop

| Masalah | Solusi |
| -------- | -------- |
| "Authentication failed" | Logout & login ulang di File → Options → Accounts |
| "Failed to push" | Klik "Pull origin" dulu untuk sync, baru push |
| "Can't commit" (file grayed out) | File ada di .gitignore. Edit .gitignore untuk include |
| Melihat `dagnostics.log` penuh | Klik Help → Show Logs Folder, hapus old logs |
| Conflict saat pull | Buka file conflict di editor, resolve manual, commit |

**Catatan**: GitHub Desktop user-friendly tapi buat task kompleks (rebase, cherry-pick), gunakan command line atau Git GUI lainnya.

### Jalankan Notebook

```bash
# Buka notebook di VS Code atau Jupyter
jupyter notebook mlops_plan_krom_bank.ipynb
```

Jalankan cell secara berurutan (1 → 13):

- Cell 1-2: Setup & imports
- Cell 3-7: Load data, EDA, preprocessing, features
- Cell 8-9: Split & baseline models
- Cell 10-12: Deep learning models & MLflow
- Cell 13: Langkah selanjutnya

### Troubleshooting Instalasi Package

| Masalah | Solusi |
| -------- | -------- |
| `ModuleNotFoundError: No module named 'statsmodels'` | Jalankan: `.venv\Scripts\python.exe -m pip install statsmodels` |
| `No module named 'tensorflow'` | TensorFlow mungkin memakan 5-10 menit install. Jika gagal, coba: `pip install tensorflow --no-cache-dir` |
| `pip command not found` | Aktifkan venv terlebih dahulu: `.venv\Scripts\activate` |
| `Permission denied` saat install package | Jalankan terminal sebagai Administrator, atau coba: `pip install --user <package>` |
| Installation timeout | Gunakan: `pip install --default-timeout=1000 <package>` |
| Disk space penuh saat install TensorFlow | Butuh ~2GB. Bebaskan space dan coba lagi. |

**Quick fix untuk semua package:**

```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --no-cache-dir pandas==3.0.1 numpy==2.4.2 matplotlib==3.10.8 seaborn==0.13.2 scikit-learn==1.8.0 statsmodels==0.15.0 tensorflow==2.20.0 mlflow==3.9.0 prophet==1.3.0 optuna==4.7.0 joblib==1.5.3 pandera==0.29.0 lightgbm==4.6.0
```

---

## **Rekomendasi Versi Library (Teruji — Python 3.13)**

Versi library yang telah diuji pada lingkungan Python 3.13 (diterapkan pada repository ini per 2026-02-13):

| Library | Versi Teruji | Catatan |
| --------- | -------------- | --------- |
| Python | 3.13.x | Target project; jangan gunakan 3.14+ untuk TensorFlow saat ini |
| pip | 26.0.1 | Upgrade via `pip install --upgrade pip |
| pandas | 3.0.1 | Digunakan untuk data handling di notebook |
| numpy | 2.4.2 | Dependensi numerik utama |
| matplotlib | 3.10.8 | Visualisasi |
| seaborn | 0.13.2 | Plotting statistik |
| scikit-learn | 1.8.0 | Algoritma ML klasik |
| statsmodels | 0.15.0 | Analisis time-series/statistik |
| tensorflow | 2.20.0 | Kompatibel dengan Python 3.13 (sebagai rilis terbaru) |
| keras | 3.0.0 | Disertakan dalam TensorFlow |
| mlflow | 3.9.0 | Experiment tracking lokal |
| prophet | 1.3.0 | Time-series forecasting |
| optuna | 4.7.0 | Hyperparameter tuning |
| joblib | 1.5.3 | Serialisasi model |
| pandera | 0.29.0 | Data validation |
| lightgbm | 4.6.0 | Gradient boosting |

### Instalasi Versi Teruji (Python 3.13)

Gunakan perintah ini untuk menginstal versi-versi yang telah diuji pada lingkungan proyek:

```bash
# Buat dan aktifkan venv (Windows)
py -3.13 -m venv .venv
.venv\\Scripts\\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install versi-versi teruji
python -m pip install --no-cache-dir \
   pandas==3.0.1 numpy==2.4.2 matplotlib==3.10.8 seaborn==0.13.2 \
   scikit-learn==1.8.0 statsmodels==0.15.0 tensorflow==2.20.0 \
   mlflow==3.9.0 prophet==1.3.0 optuna==4.7.0 joblib==1.5.3 pandera==0.29.0 lightgbm==4.6.0
```

### Versi GPU (CUDA 12.x)

Jika Anda menggunakan GPU dengan CUDA 12.x, gunakan rilis TensorFlow CUDA-aware yang cocok dengan CUDA pada mesin Anda. Contoh (pilih sesuai dokumentasi TensorFlow saat ini):

```bash
# pip (jika tersedia build yang mengemas CUDA)
python -m pip install tensorflow[and-cuda]==2.20.0

# Atau gunakan conda-forge builds yang cocok
conda install -c conda-forge tensorflow==2.20.0
```

### Compatibility Matrix (ringkasan)

| Python | TensorFlow | MLflow | pandas | scikit-learn | Status |
| -------- | -----------: | --------: | -------: | ------------: | :------: |
| 3.11 | 2.16.x | 2.x | 2.x | 1.5.x | ✅ Tested historically |
| 3.12 | 2.17.x | 3.x | 2.2.x | 1.8.x | ✅ Many packages supported |
| **3.13** | **2.20.0** | **3.9.0** | **3.0.1** | **1.8.0** | **✅ Recommended (this repo)** |
| 3.14 | ❌ Not supported (TensorFlow incompatible) | 3.x | 2.x | 1.8.x | ❌ Avoid for this project |

---

## �📈 Metrik & Evaluasi

Model dievaluasi menggunakan:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

Pembagian Train/Val/Test: **70% / 15% / 15%** (time-series aware)

---

## 💡 Tips Pengembangan (Dev Tips)

### 🖥️ Backend (Model Serving)

1. **Validasi Data**: Gunakan `pydantic` pada FastAPI untuk memvalidasi input data saham (misal: format tanggal, range harga) sebelum dikirim ke model.
2. **Async Operations**: Manfaatkan `async def` untuk endpoint API guna menangani beban request yang tinggi tanpa memblokir thread.
3. **Model Caching**: Muat model (misal `.pkl` atau `.h5`) sekali saat startup aplikasi, jangan memuat balik untuk setiap request prediksi.
4. **Versioning**: Berikan prefix versi pada URL API (contoh: `/api/v1/predict`) untuk mempermudah migrasi jika arsitektur model berubah.

### 🎨 Frontend (Dashboard/UI)

1. **Real-time Charts**: Gunakan library seperti `Plotly.js` atau `Chart.js` untuk membuat grafik harga saham yang interaktif dan responsif.
2. **Error Handling**: Implementasikan fallback UI (seperti skeleton loader atau pesan error yang ramah) saat API backend sedang memproses data atau jika model gagal memproses input.
3. **State Management**: Simpan hasil prediksi sementara di state (seperti React Context atau Vuex) agar user tidak perlu melakukan request berulang untuk data yang sama.
4. **Mobile First**: Pastikan dashboard metriks tetap terbaca jelas di perangkat mobile, terutama grafik time-series yang panjang.

---

## 📝 Catatan

- **Versi Python**: Project menargetkan **Python 3.13 maksimal**. TensorFlow & MLflow tidak kompatibel dengan Python 3.14+. Gunakan `py -3.13 -m venv .venv` untuk memastikan versi yang benar.
- **TensorFlow & Prophet**: Membutuhkan package besar; kecepatan instalasi sangat bergantung pada kualitas internet Anda.
- **Instalasi Gagal?**: Jika `pip` terputus karena timeout, gunakan `--default-timeout=1000` atau cek bagian Troubleshooting.
- **Error [WinError 32]**: Jika muncul pesan "process cannot access the file", biasanya karena Kernel Jupyter sedang aktif. **Restart Kernel** atau tutup notebook sebelum instalasi.
- **Dataset**: Diunduh dari Kaggle; pastikan kredensial valid di `~/.kaggle/kaggle.json`
- **MLflow**: Tracking lokal di `artifacts/mlruns/`; bisa dilihat via `mlflow ui`
- **Security**: Jangan commit `kaggle.json` atau credentials ke repository

---

## 📞 Kontak & Referensi

- **Dataset**: [Harga Saham Historis KROM Bank Indonesia](https://www.kaggle.com/datasets/caesarmario/krom-bank-indonesia-stock-historical-price)
- **Libraries**: TensorFlow, scikit-learn, pandas, MLflow, Prophet
- **Best Practices MLOps**: Model versioning, experiment tracking, automated retraining

---

**Terakhir Diperbarui**: 14 Februari 2026
