README ini bertujuan untuk menjelaskan langkah-langkah yang dijalankan pada TP ini
Notes: beberapa comment pada kode dibantu oleh chatGPT :D

File TP2 digunakan untuk meretrieve TOP-100 document yang relevan:
- bsbi.py
- compression.py
- util.py
- index.py

File TP3 digunakan untuk melakukan reranking document, 
dengan langkah-langkah sebagai berikut:

1. File data_preparator.py

File ini berisi function function yang dapat membantu kita untuk melakukan 
preparation data. Di dalam file ini ada beberapa fungsi seperti melakukan
mapping, memproses query dan documents, menyimpan dan load file pickle, 
membuat training dan testing dataset, serta mempersiapkan qrels

Ketika dijalankan program ini akan menyiapkan mapping document dan query
yang telah diproses dan disimpan dalam pickle

2. File retrieval.py
File ini ditujukan untuk mempersiapkan testing dataset, di mana pada kode ini
akan ada beebrapa fungsi untuk me-retrieve top 100 document dan diubah formatnya
untuk persiapan data

Ketika dijalankan program ini akan membuat dataset dari top-100 doc yang diretrieve

3. File ranker.py
File ini digunakan untuk membuat, melatih, dan menggunakan model ranker 
berdasarkan representasi vektor dengan pendekatan LSI (Latent Semantic Indexing) 
dan model LightGBM.

Ketika dijalankan program ini akan melatih model dan juga mengevaluasinya pada
data validation.

4. File letor.py
File ini digunakan untuk melakukan pengujian letor pada data testing. Pada kode
ini juga akan dilakukan evaluasi pada hasil reranking pada data test.

Ketika dijalankan program ini akan melakukan prediksi pada data testing dan
mengevaluasinya. Lalu, hasil rerankingnya akan disimpan

[Bonus]

-------------------------------------------------------------------
* Folder pickle digunakan untuk menyimpan data data hasil processing sebelumnya
* File csv merupakaan hasil dari reranking pada data testing 
  (setelah menggunakan letor atau masih bsbi biasa)
* lgbr_base.txt merupakan model ranker yang disimpan
* lsi_base merupakan model LSI yang sudah dilatih
* tfidf_vectorizer merupakan model TFIDF yang sudah dilatih

* coba-coba.ipynb merupakan file untuk coba coba dan debugging saya :D