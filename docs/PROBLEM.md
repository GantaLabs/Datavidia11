# Problem Specification: Datavidia-11-0

## 1. Overview & Problem Statement
Overview
Setiap paket yang dikirim meninggalkan jejak. Paket dipindai saat dijemput, disortir di hub, berpindah moda di tengah jalan, lalu dipindai lagi ketika kurir mengetuk pintu penerima. Tagihan terbit, promo terpakai, dan catatan alamat ditambahkan. Bagi orang yang sudah lama di lapangan, jejak itu sering cukup untuk menebak nasib sebuah paket: sampai dengan mulus, tertahan di gudang, atau pulang lagi ke pengirimnya. Rian, kepala tim pengendalian mutu di sebuah jaringan ekspedisi antarpulau, menatap layar yang penuh baris merah. Status penanganan setiap pengiriman selama ini diisi manual oleh petugas di akhir proses. Petugasnya terlambat mengisi, lupa mengisi, atau terburu-buru memilih status yang paling mudah. Akibatnya laporan bulanan tidak bisa dipercaya. Tim klaim tidak tahu paket mana yang benar-benar hilang, dan tim operasional tidak tahu hub mana yang paling sering menahan barang. Rian punya data operasional lengkap yang berisi lebih dari 1,6 juta pengiriman beserta pelanggan, kurir, hub, jenis layanan, promo, log pemindaian, kaki rute, dan tagihan. Ia ingin sebuah model yang dapat membaca jejak itu dan menandai status penanganan setiap pengiriman secara otomatis.

Tugas ini tidak sederhana. Datanya tersebar di sembilan tabel yang saling berelasi. Satu pengiriman bisa punya belasan baris pemindaian dan beberapa kaki rute. Kelas-kelas yang paling penting, seperti kerusakan dan klaim kehilangan, justru paling jarang muncul. Sebagian catatan juga tidak lengkap, dan tidak semua kolom benar-benar berguna. Model perlu cukup jeli untuk mengenali kasus yang jarang, tanpa mengorbankan ketepatan pada kasus yang umum. Keputusan akhir tetap di tangan petugas; model hanya membantu mereka memeriksa pengiriman yang tepat lebih dulu.

Description
Dalam kompetisi ini, peserta diminta membangun model klasifikasi untuk menentukan status penanganan setiap pengiriman. Untuk setiap shipment_id di test/shipments.csv, model harus menghasilkan satu label status_penanganan.

---

Setiap pengiriman termasuk ke dalam tepat satu dari tujuh status berikut.
Kelas			        Deskripsi
normal			        Pengiriman selesai tanpa insiden
konfirmasi_alamat	    Pengiriman membutuhkan konfirmasi atau perbaikan alamat sebelum dapat diselesaikan
tertunda_operasional	Pengiriman tertunda karena kendala operasional, misalnya penumpukan di hub atau keterlambatan transit
gagal_antar_awal	    Pengiriman gagal diantar pada percobaan pertama
retur_pengirim		    Paket dikembalikan ke pengirim
rusak_perjalanan	    Paket mengalami kerusakan selama perjalanan
klaim_hilang		    Paket hilang dan berakhir sebagai klaim

---
Evaluation :
Tujuan (Goal)
Untuk setiap shipment_id pada sample_submission.csv, prediksikan satu nilai status_penanganan, yaitu salah satu dari tujuh kelas di atas.

---
Metrik (Metric) :
Submission dinilai dengan Macro F1-score (1/7 F1k)

Deskripsi Data
Data kompetisi adalah cuplikan data operasional sebuah jaringan ekspedisi: 1.605.862 pengiriman beserta pelanggan, kurir, hub, jenis layanan, promo, log pemindaian, kaki rute, dan tagihan. Data dibagi ke dalam sembilan tabel CSV yang saling berelasi lewat kolom ID, seperti sebuah basis data kecil.

File
train/shipments.csv - the training set: pengiriman dengan kolom target status_penanganan
test/shipments.csv - the test set: pengiriman tanpa kolom target
train/scan_events.csv, test/scan_events.csv - log pemindaian per pengiriman
train/route_segments.csv, test/route_segments.csv - kaki rute per pengiriman
train/invoices.csv, test/invoices.csv - tagihan pengiriman dan langganan
customers.csv - data pelanggan
couriers.csv - data kurir
hubs.csv - data hub sortasi dan cabang
service_packages.csv - jenis layanan pengiriman
promotions.csv - program promo
sample_submission.csv - a sample submission file in the correct format




---

## 2. Technical Task Specifications
- **Task Type:** `multiclass_classification`
- **Target Variable(s):** `['status_penanganan']` (Primary: `status_penanganan`, Paired Entities: `False`)
- **ID Column:** `shipment_id`
- **Official Evaluation Metric:** `f1_macro` (Optimization: Higher is Better)
- **Locked Validation Protocol:** `stratified` (`data/processed/cv_splits.json`)
- **Temporal Split:** `False`
- **Adversarial Validation AUC (Train vs Test):** `0.5154`
- **Drifting Features Warning (Avoid raw monotonic/ID use):** `['customer_id', 'courier_id', 'hub_id', 'package_id', 'promo_id', 'cour_hub_id']`
- **Demographic Shift Warnings:** `[]`

---

## 3. Directory Layout
- **Train Data:** `data/raw/train.csv`
- **Test Data:** `data/raw/test.csv`
- **Sample Submission:** `data/raw/sample_submission.csv`
- **Locked CV Splits:** `data/processed/cv_splits.json`
- **Dataset Profile:** `configs/data_profile.json`
- **Auxiliary Data:** Stored in `data/raw/` subdirectories
