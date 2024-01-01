Link deployment: [https://brent-fe.vercel.app/](https://brent-fe.vercel.app/)

Home Page:

<img src="https://github.com/bryan273/brent-be/assets/88226713/021ccd6b-e00e-4ce1-8973-609ca1477e03" width="600" height="255"/>

Search Page:

<img width="600" alt="brent search amnesia" src="https://github.com/bryan273/brent-be/assets/88226713/56e78da1-79cc-45ef-95cb-48fd62bb2ef0">

<img width="600" alt="brent search essential tremor" src="https://github.com/bryan273/brent-be/assets/88226713/911f1923-e6ee-40bd-809e-bb409db0f30e">

Searched Document:

<img width="600" alt="doc result essential tremor" src="https://github.com/bryan273/brent-be/assets/88226713/74c47e5a-d044-403b-b23d-953ec30c5410">

File TP2 digunakan untuk meretrieve TOP-N document yang relevan:
- bsbi.py
- compression.py
- util.py
- index.py

File TP4 digunakan untuk melakukan reranking document, 
dengan langkah-langkah sebagai berikut (lengkapnya ada di branch full):

1. File data_preparator.py

File ini berisi function function yang dapat membantu kita untuk melakukan 
preparation data. Di dalam file ini ada beberapa fungsi seperti melakukan
mapping, memproses query dan documents, menyimpan dan load file pickle, 
membuat training dan testing dataset, serta mempersiapkan qrels

Ketika dijalankan program ini akan menyiapkan mapping document dan query
yang telah diproses dan disimpan dalam pickle

2. File retrieval.py
File ini ditujukan untuk mempersiapkan testing dataset, di mana pada kode ini
akan ada beebrapa fungsi untuk me-retrieve top N document dan diubah formatnya
untuk persiapan data

Ketika dijalankan program ini akan membuat dataset dari top-N doc yang diretrieve

3. File ranker.py
File ini digunakan untuk membuat, melatih, dan menggunakan model ranker 
berdasarkan representasi vektor dari BERT dan model LightGBM.

Ketika dijalankan program ini akan melatih model dan juga mengevaluasinya pada
data validation.

4. File letor.py
File ini digunakan untuk melakukan pengujian letor pada data testing. Pada kode
ini juga akan dilakukan evaluasi pada hasil reranking pada data test.

Ketika dijalankan program ini akan melakukan prediksi pada data testing dan
mengevaluasinya. Lalu, hasil rerankingnya akan disimpan

-------------------------------------------------------------------
* Folder pickle digunakan untuk menyimpan data data hasil processing sebelumnya
* File csv merupakaan hasil dari reranking pada data testing 
  (setelah menggunakan letor atau masih bsbi biasa)
* lgbr_base.txt merupakan model ranker yang disimpan

* coba-coba.ipynb merupakan file untuk coba coba dan debugging saya :D
