# YOLOv8 Tabanlı Derin Öğrenme Yaklaşımlarıyla Beyin Tümörü Tespiti 🧠

Bu depo, **İstanbul Topkapı Üniversitesi Derin Öğrenme Dersi Final Ödevi** kapsamında geliştirilen beyin tümörü tespiti projesinin kodlarını, modellerini ve analiz sonuçlarını içermektedir.

## 📌 Proje Amacı
Manyetik rezonans (MR) görüntülerinden beyin tümörlerinin otomatik, hızlı ve yüksek doğrulukla tespit edilmesini sağlayacak derin öğrenme tabanlı bir nesne tespiti (object detection) modeli geliştirmektir. Projede, son teknoloji YOLOv8 (Nano) mimarisi kullanılarak transfer öğrenme ve hiperparametre optimizasyonu stratejileri uygulanmıştır.

## 📊 Veri Seti
Projeye temel oluşturan MR görüntüleri, Ultralytics tarafından sağlanan [Brain Tumor Detection](https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip) veri setinden elde edilmiştir. 
* **Sınıflar:** `0 - negative` (Sağlıklı/Tümörsüz), `1 - positive` (Tümörlü)
* **Veri Dağılımı:** 893 Eğitim (Train) görüntüsü, 223 Doğrulama (Validation) görüntüsü.

## 🛠️ Yöntem ve Geliştirme Ortamı
* **Çalışma Ortamı:** Google Colab (Tesla T4 GPU)
* **Model Mimarisi:** YOLOv8n (Nano) önceden eğitilmiş ağırlıklarla (Transfer Learning).
* **Görüntü Boyutu:** 640x640 piksel.

Proje kapsamında iki farklı yaklaşım test edilmiş ve kıyaslanmıştır:
1. **Baseline Model (Temel):** Proje dokümanında istenen varsayılan ayarlar (50 Epoch, Varsayılan Veri Artırımı).
2. **Optimize Edilmiş Model:** Tıbbi MR görüntülerinin anatomik bütünlüğünü bozmamak adına `mosaic=0.0` (mozaik veri artırımı kapalı) ayarı ve daha derin özellik çıkarımı için `100 Epoch` ile eğitilen gelişmiş model.

## 📈 Sonuçlar ve Performans Analizi
Optimize edilmiş modelin başarım metrikleri aşağıdaki gibidir:

* **Genel Ortalama Hassasiyet (mAP@0.5):** %53.6
* **Duyarlılık (Recall - Tümör Yakalama Oranı):** %93.1 
* **Genel Doğruluk (Accuracy):** %53.6

**Mühendislik Çıkarımı:** Model optimizasyonu sonucunda, tümörleri tespit etme (Recall) oranından ödün vermeden, sağlıklı dokuları yanlışlıkla tümör sanma (False Positive) sayısında azalma sağlanmıştır. YOLOv8n modelinin sınırlı kapasitesi göz önüne alındığında, modelin tümörleri gözden kaçırmama konusundaki başarısı klinik açıdan anlamlı bulunmuştur.

## 📂 Depo İçeriği
* `YOLOv8_Tabanlı_Derin_Öğrenme_Yaklaşımlarıyla_Beyin_Tümörü_Tespiti.ipynb` : Projenin tüm eğitim, test ve optimizasyon adımlarını içeren kaynak kod dosyası.
* `best.pt` : 100 Epoch sonucunda elde edilen, en yüksek başarıma sahip optimize edilmiş model ağırlıkları.
* `confusion_matrix.png` : Modelin hata ve doğruluk dağılımını gösteren karışıklık matrisi.
* `results.png` : Eğitim sürecine ait kayıp (loss) ve başarım (mAP, Precision, Recall) eğrileri.

## 🚀 Nasıl Çalıştırılır?
Modeli kendi ortamınızda test etmek için aşağıdaki Python kodunu kullanabilirsiniz:

```python
from ultralytics import YOLO

# Eğitilen modeli yükle
model = YOLO("best.pt")

# Yeni bir MR görüntüsü üzerinde tahmin yap
results = model.predict(source="test_resmi.jpg", save=True, conf=0.25)
