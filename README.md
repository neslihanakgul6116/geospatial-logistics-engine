# 🚀 Donanım Hızlandırmalı Coğrafi Lojistik Motoru
### SIMD, GPU (CUDA) ve SSD Destekli Büyük Veri Mesafe Optimizasyon Sistemi

---

## 📌 Projenin Amacı
Bu proje; lojistik, kargo ve e-ticaret ağlarında milyonlarca müşteri noktası ile merkez depolar arasındaki mesafeleri **milisaniyeler içinde** hesaplayarak en yakın dağıtım rotalarını çıkaran yüksek performanslı bir simülasyon motorudur. 

Klasik yazılımlarda kullanılan standart `for` döngüleri, 10 milyonlarca veriyi sırayla işlediği için sistemleri kilitlemektedir. Bu projede ise **donanımsal vektörizasyon (SIMD), grafik işlemci (GPU) paralelleştirmesi ve akışlı veri yönetimi** kullanılarak bu dezavantaj ortadan kaldırılmış ve hesaplama hızı saniyenin altına indirilmiştir.

---

## ⚙️ Teknik Altyapı ve Donanım Optimizasyonları

Projenin arkasındaki mühendislik gücü üç temel sütun üzerine kuruludur:

1. **CPU Tabanlı SIMD (Tek Komut, Çoklu Veri) Vektörizasyonu:**
   * Python döngüleri yerine **NumPy** kütüphanesi kullanılmıştır. Veriler RAM'e ardışık olarak dizilir ve işlemcinin alt seviye SIMD komut setleri tetiklenerek trigonometrik Haversine formülü milyonlarca satıra **aynı anda (tek hamlede)** uygulanır.
2. **GPU (CUDA / CuPy) Hızlandırma Desteği:**
   * Sistem, bilgisayarda NVIDIA ekran kartı (CUDA) varsa hesaplamaları doğrudan GPU'nun binlerce paralel çekirdeğine devreder (`CuPy`). Donanım yoksa otomatik olarak CPU moduna geçer (Akıllı Fallback mekanizması).
3. **Büyük Veri (Big Data) ve SSD Akış Optimizasyonu:**
   * 10 milyon ve üzeri kayıtlarda RAM sınırlarının aşılmaması için **Polars** altyapısı kullanılmıştır. Veriler bellek dostu yöntemlerle yönetilerek disk/RAM akışı optimize edilmiştir.

---

## 🗺️ Temel Özellikler ve Arayüz

* **Dinamik Depo Yönetimi:** Türkiye'nin ticaret ve lojistik merkezleri olan **İstanbul, Ankara, İzmir, Bursa ve Kocaeli** illerinden biri ana depo üssü olarak seçilebilir.
* **Modern Web Paneli:** **FastAPI** arka plan motoruyla güçlendirilmiş, **Tailwind CSS** arayüzüne sahip şık bir kontrol paneli sunar.
* **İnteraktif Harita:** **Leaflet.js** ve OpenStreetMap entegrasyonu ile deponun konumu (kırmızı) ve en yakın 5 müşteri noktası (mavi) harita üzerinde anlık olarak görselleştirilir.

---

## 🛠️ Kurulum ve Çalıştırma Rehberi

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

### 1. Gerekli Kütüphanelerin Yüklenmesi
Terminal penceresini açın ve şu komutu çalıştırın:
```bash
python -m pip install fastapi uvicorn polars numpy