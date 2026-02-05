# Cara Mendapatkan dan Menaruh `kaggle.json`

Langkah singkat (dalam Bahasa Indonesia):

1. Masuk ke akun Kaggle Anda: https://www.kaggle.com/
2. Klik profil -> Account -> Scroll ke bagian 'API' -> klik 'Create New API Token'. File `kaggle.json` akan diunduh.
3. Tempatkan file `kaggle.json` tersebut di salah satu lokasi berikut:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

Alternatif (tanpa menyimpan file): set environment variables:

- PowerShell (sesi saat ini):
  $env:KAGGLE_USERNAME = "your_username"
  $env:KAGGLE_KEY = "your_key"

- Permanen (Windows):
  setx KAGGLE_USERNAME "your_username"
  setx KAGGLE_KEY "your_key"

Keamanan & Git:
- **Jangan** commit `kaggle.json` ke repositori publik.
- File template `kaggle.json.template` ada di root proyek; ganti nilainya, lalu jalankan `.\setup_kaggle.ps1` untuk menyalin ke `%USERPROFILE%\.kaggle\kaggle.json`.

Otomatisasi unduhan dataset (contoh):

1. Install Kaggle CLI: `pip install kaggle`
2. Jalankan (setelah kredensial siap):
   `kaggle datasets download -d caesarmario/krom-bank-indonesia-stock-historical-price -p dataset --unzip`

Jika Anda ingin, saya bisa membantu memindahkan file template menjadi `kaggle.json` setelah Anda memberikan kredensial atau mengisi file template secara lokal. Saya tidak dapat membuat atau mengisi kredensial asli untuk Anda. 
**PENTING (Keamanan):** Jika kredensial Anda sempat terekspos di repositori, segera lakukan rotasi API key di Kaggle (hapus yang lama dan buat kunci baru). Selain itu, hapus kredensial dari riwayat Git (mis. menggunakan `git filter-repo` atau BFG) dan pastikan kredensial tidak tersimpan di repositori publik.