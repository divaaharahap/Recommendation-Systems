# Anime Recommendation System
Rekomendasi Sistem — Sistem Rekomendasi berbasis Content-Based Filtering & Collaborative Filtering

## Project Overview
Sistem rekomendasi telah menjadi komponen vital dalam berbagai platform digital, termasuk e-commerce, layanan streaming, dan komunitas daring. Dalam konteks hiburan seperti anime, pengguna dihadapkan pada ribuan judul dengan genre, tipe, dan episode yang beragam. Tanpa bantuan sistem penyaring, pengguna sering kali mengalami kesulitan dalam menemukan tontonan yang sesuai dengan preferensi mereka. Oleh karena itu, pengembangan anime recommender system menjadi penting untuk meningkatkan pengalaman pengguna melalui personalisasi.

Proyek ini secara khusus memanfaatan dua pendekatan utama dalam sistem rekomendasi, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF). Pendekatan CBF memungkinkan sistem untuk menyarankan anime yang mirip dengan yang sebelumnya disukai oleh pengguna berdasarkan fitur konten seperti genre. Sementara itu, CF memanfaatkan pola interaksi antar pengguna untuk memberikan rekomendasi yang lebih personal.

Relevansi dan urgensi proyek ini diperkuat oleh data dari platform MyAnimeList.net, di mana ribuan pengguna memberikan rating terhadap anime. Dataset tersebut yang telah dibagikan secara terbuka di Kaggle, digunakan dalam proyek ini. Penelitian sebelumnya juga menunjukkan bahwa sistem rekomendasi berbasis hybrid--gabungan antara CBF dan CF-- dapat meningkatkan akurasi dan kepuasan pengguna secara signifikan (Bobadilla et al., 2013; Aggarwal, 2016).

Dengan demikian, proyek ini tidak hanya relevan dalam konteks praktis dunia hiburan digital, tetapi juga mendemonstrasikan penerapan teknik machine learning dalam menyelesaikan masalah nyata di bidang personalisasi informasi.

## Business Understanding
### Problem Statements
- Pengguna kesulitan menemukan anime baru yang sesuai dengan preferensi mereka.
- Banyaknya judul anime membuat pengguna bingung memilih tontonan.

### Goals
- Membantu pengguna menemukan anime baru yang relevan dan sesuai preferensi mereka.
- Mengembangkan sistem rekomendasi yang dapat memberikan top-N rekomendasi anime yang relevan dan bervariasi.

### Solution Approach
- Content-Based Filtering <br>
Rekomendasi berdasarkan kemiripan fitur konten (genre, rating, dll) dari anime yang sudah disukai pengguna.
- Collaborative Filtering <br>
Rekomendasi berdasarkan interaksi user-user lain yang memiliki preferensi mirip.

## Data Understanding
- Dataset didapat dari Kaggle.
- Link Kaggle : https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- Dataset ini berisi preferensi pengguna terhadap anime, yang diambil dari MyAnimeList.net
- Terdiri dari dua file utama:
<br>

anime.csv - Metadata Anime :
- anime_id :	ID unik anime dari MyAnimeList
- name : Judul lengkap anime
- genre : Daftar genre (dipisah koma)
- type : Jenis tayangan (TV, Movie, OVA, dll)
- episodes : Jumlah episode (1 jika film)
- rating : Rata-rata rating komunitas (out of 10)
- members :	Jumlah member yang memiliki anime ini di daftar mereka
<br>

rating.csv - interaksi pengguna
- user_id : id acak pengguna
- anime_id : id anime yang dirating
- rating : rating pengguna
<br>

- Dataset `anime.csv` memiliki 12.294 baris, yang berarti terdapat 12.294 anime unik dan memiliki 7 kolom fitur.
- Dataset `rating.csv` memiliki 7.813.737 baris, artinya ada hampir 8 juta interaksi user terhadap anime dan memiliki 3 kolom fitur
<br>

Tipe data `anime.csv` <br>
Kolom yang bertipe numerik
- anime_id
- rating
- members
Kolom yang bertipe object
- name
- genre
- type
- episodes

<br>

Tipe data `rating.csv`<br>
Semua kolom bertipe data numerik (user_id, anime_id, rating)

### Univariate EDA
- Melakukan `describe()` di `anime.csv` dengan kesimpulan:
- Banyak anime yang belum dirating (rating<count)
- Rating komunitas rata-rata tinggi
- Penyebearan member sangat timpang

- Memfilter data yang valid (rating > 0) dan memvisualisasikanya untuk melihat distribusi rating user.<br>
![Distribusi rating user](images/image.png)<br>
**Insight**
User cenderung memberi rating tinggi, sehingga sistem harus hati-hati agar tidak menganggap semua anime bagus hanya karena rating tinggi.

- Visualisasi anime dengan members terbanyak <br>
![Anime Favorit](images/image-1.png)<br>

- Melakukan `describe()` di `rating.csv`
Dengan insight yang didapat adalah data cenderung positif yang berarti user lebih sering memberi rating tinggi.

## Data Preparation
- Mengganti nama kolom rating di anime.csv menjadi mean_rating untuk menghindari duplikasi nama kolom saat penggabungan.
- Menangani missing values :
1. Menghapus baris dengan nama anime kosong.
2. Mengisi genre, type yang kosong dengan nilai default.
3. Menghapus baris dengan episode yang tidak diketahui (unknown)
4. Mengisi nilai mean_rating dengan rata-rata dan members dengan median.
![Akhir missing values](images/image-3.png) <br>
- Menghapus data duplikat berdasarkan keseluruhan dan berdasarkan pasangan user_id dan name.
- Mengonversi kolom episodes menjadi tipe numerik.
### Content-Based Filtering
- Menggabungkan fitur genre dan type menjadi kolom content untuk representasi teks yang akan digunakan pada teknik TF-IDF.
- Vektorisasi Teks dengan TF-IDF <br>
```
tfidf = TfidfVectorizer()
tfidf.fit(df_model['content'])
tfidf.get_feature_names_out()
```
Digunakan untuk mengukur pentingnya sebuah kata dalam dokumen relatif terhadap seluruh korpus. Ini mengurangi bobot kata-kata umum dan memperkuat kata-kata khas dalam deskripsi anime.

- Transformasi ke Bentuk Matriks <br>
```
tfidf_matrix.todense()
```
Matriks TF-IDF dikonversi ke bentuk densitas untuk mempermudahkan pemetaan ke DataFrame.

- Pembuatan DataFrame representasi TF-IDF.
```
tfidf_df = pd.DataFrame(
    tfidf_matrix.todense().round(2),
    columns=tfidf.get_feature_names_out(),
    index=df_model['name']
)
```
- Melakukan vektorisasi content menggunakan TF-IDF Vectorizer (penjelasan teknis ini masuk ke Data Preparation).

### Collaborative Filtering
- Menghapus rating yang dibawah 1, rating dibawah 1 menjelaskan bahwa user tidak melakukan rating untuk anime yang sudah selesai ditonton, dan itu tidak diperlukan di proses ini.
- Mengacak baris dataset (shuffling) dilakukan agar data train-val terdistribusi acak dan tidak terurut berdasarkan pola tertentu.
- Melakukan encoding user_id dan anime_id menjadi angka. Encoding ID penting karena model hanya bisa memahami input numerik, bukan string atau ID aslinya.
- Menyusun kamus pemetaan dibuat agar bisa mengubah kembali hasil prediksi ke label asli.
- Mengubah tipe data rating ke float, ini penting agar model bisa belajar dengan skala yang seragam dan mempercepat konvergensi.
- Normalisasi nilai rating ke skala 0–1 ini untuk membuat data dalam format yang dapat diproses oleh jaringan neural.
- Membagi dataset menjadi data pelatihan dan validasi, membantu evaluasi model dengan data yang tidak pernah dilihat selama training.

## Modelling
### Content-Based Filtering
Pendekatan Content-Based Filtering dalam proyek ini dilakukan dengan mengubah fitur content menjadi representasi numerik menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency). Fitur content merupakan gabungan informasi seperti genre, tipe, dan rating yang mencerminkan karakteristik unik dari setiap anime.

Setelah representasi TF-IDF terbentuk, sistem menghitung kemiripan antar anime menggunakan cosine similarity untuk mengukur seberapa mirip dua anime berdasarkan fitur konten mereka. Model kemudian merekomendasikan anime yang memiliki skor similarity tertinggi terhadap anime yang sudah disukai pengguna.

#### Langkah-langkah modelling
1. Menyiapkan data content sebagai gabungan fitur genre, tipe, dan rating.
2. Melakukan transformasi teks ke numerik menggunakan TF-IDF vectorizer.
3. Menghitung cosine similarity antar anime berdasarkan matriks TF-IDF.
- Penghitungan kemiripan antar anime.
```
cosine_sim = cosine_similarity(tfidf_matrix)
```
5. Mengembangkan fungsi rekomendasi untuk menghasilkan daftar Top-N anime yang paling mirip berdasarkan konten.

Contoh output Top-10 rekomendasi Content-Based Filtering untuk anime “Naruto”:
![image](https://github.com/user-attachments/assets/dc081449-b373-4e4e-acae-71b9ce53e239)

### Collaborative Filtering
Model Collaborative Filtering dibangun dengan menggunakan neural network yang terdiri dari dua embedding layer untuk memetakan user_id dan anime_id ke dalam vektor berdimensi 50. Vektor embedding ini digabungkan melalui operasi dot product untuk memprediksi rating pengguna terhadap sebuah anime.

Model dilatih menggunakan fungsi loss Mean Squared Error (MSE) dan optimizer Adam. Struktur model ini sederhana namun efektif dalam menangkap pola interaksi pengguna dan preferensi rating yang bersifat personal.
<br>

#### Langkah-langkah modelling
1. Melakukan encoding pada user_id dan anime_id agar dapat diproses oleh model.
2. Membuat embedding layer untuk masing-masing entitas dengan dimensi 50.
3. Menggabungkan embedding dengan dot product untuk mendapatkan prediksi rating.
4. Melatih model menggunakan MSE loss dan optimizer Adam selama beberapa epoch.
5. Memvalidasi performa model menggunakan data validasi.
6. Membuat fungsi rekomendasi Top-N anime dengan rating prediksi tertinggi untuk setiap user.
Output Top-5 rekomendasi Collaborative Filtering untuk user dengan ID 100:
![image](https://github.com/user-attachments/assets/a3728e0a-b750-4cf8-81ac-6bf77f673a31)


Parameter yang digunakan :
| Komponen          | Nilai                           |
|-------------------|---------------------------------|
| Dimensi Embedding | 50                              |
| Fungsi Aktivasi   | Tidak digunakan (hanya dot product) |
| Loss Function     | Mean Squared Error (MSE)        |
| Optimizer         | Adam, learning_rate=0.001       |
| Regularizer       | L2 (lambda = 1e-6)              |

#### Training Model
Model dilatih selama 5 epoch dengan `batch_size = 4096` untuk mempersingkat waktu. Data dibagi menjadi 80% training dan 20% validation untuk mengevaluasi generalisasi model.

#### Visualisasi Kinerja
Hasil training divisualisasikan dalam bentuk grafik **Root Mean Squared Error (RMSE)** untuk data training dan validation per epoch. Dari grafik terlihat bahwa nilai RMSE menurun dan stabil, tanpa indikasi overfitting. <br>
![Root Mean Squared Error over Epochs](images/image-4.png) <br>

### Kekurangan dan Kelebihan CF dan CBF
| Metode                  | Kelebihan                                                                 | Kekurangan                                                                 |
|-------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Collaborative Filtering | - Dapat memberikan rekomendasi yang sangat personal<br>- Tidak butuh metadata item | - Cold start problem (sulit merekomendasikan untuk user/item baru)<br>- Rentan terhadap sparsity data |
| Content-Based Filtering | - Bisa bekerja meskipun user/item baru<br>- Rekomendasi berbasis fitur item | - Kurang variatif karena cenderung merekomendasikan item yang mirip<br>- Butuh data deskriptif (fitur) item yang lengkap |

## Evaluasi
### Metode Evaluasi CBF
- Hasil Evaluasi <br>
![precision@10 Naruto](images/image-6.png) <br>
Precision@10 sebesar 0.6 berarti dari 10 anime yang direkomendasikan untuk pengguna yang menyukai "Naruto", sebanyak 6 di antaranya benar-benar relevan.

### Metode Evaluasi CF
- RMSE (Root Mean Squared Error) untuk mengukur selisih antara rating yang diprediksi dan rating asli.
- Formula RMSE
```
RMSE = sqrt( (1/n) * Σ (y_i - ŷ_i)² )
```
- Hasil Evaluasi <br>
![Hasil Evaluasi](images/image-8.png)

RMSE yang rendah menunjukkan prediksi model sangat mendekati rating sebenarnya. <br>

## Kesimpulan
Proyek ini berhasil membangun sistem rekomendasi anime menggunakan dua pendekatan utama, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF). Dengan memanfaatkan dataset dari MyAnimeList.net, sistem ini mampu memberikan rekomendasi anime yang relevan berdasarkan preferensi pengguna sebelumnya maupun pola interaksi dari pengguna lain yang serupa.

Melalui teknik TF-IDF dan cosine similarity, pendekatan CBF memungkinkan sistem memahami kemiripan konten antar anime. Di sisi lain, CF berbasis matriks rating pengguna memungkinkan sistem untuk mempelajari pola preferensi yang lebih dalam dan personal. Evaluasi model menggunakan metrik RMSE menunjukkan performa model dalam memprediksi rating yang mendekati nilai aktual.

Secara keseluruhan, sistem ini membuktikan bahwa integrasi pendekatan konten dan kolaboratif dapat meningkatkan kualitas rekomendasi, sekaligus memberikan pengalaman pengguna yang lebih baik dalam menjelajahi ribuan judul anime. Proyek ini juga menjadi contoh penerapan nyata machine learning dalam dunia hiburan digital yang relevan dan bermanfaat.

## References
Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013).
Recommender Systems Survey. Knowledge-Based Systems, 46, 109–132.
