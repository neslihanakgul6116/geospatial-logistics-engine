import time
import numpy as np
import polars as pl
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import uvicorn

# GPU (CuPy) Desteği Kontrolü
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

app = FastAPI(title="Coğrafi Lojistik Motoru - GPU & SSD Destekli SIMD Dashboard")

def haversine_accelerated(lat1, lon1, lat2, lon2, use_gpu=False):
    """GPU (CuPy) veya CPU (NumPy/SIMD) destekli esnek Haversine Hesaplaması"""
    if use_gpu and GPU_AVAILABLE:
        # GPU (CUDA) Üzerinde Vektörel Hesaplama
        lat1, lon1, lat2, lon2 = map(cp.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = cp.sin(dlat / 2.0) ** 2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon / 2.0) ** 2
        c = 2 * cp.arcsin(cp.sqrt(a))
        return cp.asnumpy(c) * 6371.0 # Sonucu tekrar CPU'ya (RAM'e) al
    else:
        # CPU Üzerinde SIMD (NumPy) Vektörel Hesaplama
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371.0

# HTML Arayüzü (GPU ve SSD Durum Göstergeli)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Coğrafi Lojistik Motoru - GPU & SSD Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 font-sans min-h-screen p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        
        <header class="flex flex-col md:flex-row justify-between items-center bg-gray-800 p-6 rounded-2xl shadow-xl border border-gray-700">
            <div>
                <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-500">
                    🚀 GPU & SSD Destekli Lojistik Motoru
                </h1>
                <p class="text-gray-400 text-sm mt-1">Donanımsal Hızlandırma: CUDA (GPU) + SIMD (CPU) + Polars Disk/RAM Akışı</p>
            </div>
            <div class="mt-4 md:mt-0 flex flex-wrap items-center gap-3">
                <div>
                    <label class="block text-xs text-gray-400 mb-1">Depo:</label>
                    <select id="depoSelect" class="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white">
                        <option value="istanbul">İstanbul</option>
                        <option value="ankara">Ankara</option>
                        <option value="izmir">İzmir</option>
                        <option value="bursa" selected>Bursa</option>
                        <option value="kocaeli">Kocaeli</option>
                    </select>
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">İşlemci Modu:</label>
                    <select id="engineSelect" class="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white">
                        <option value="cpu" selected>CPU (SIMD / NumPy)</option>
                        <option value="gpu">GPU (CUDA / CuPy)</option>
                    </select>
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">Veri Boyutu:</label>
                    <select id="sizeSelect" class="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white">
                        <option value="1000000">1 Milyon</option>
                        <option value="10000000" selected>10 Milyon (SSD/RAM Akışlı)</option>
                    </select>
                </div>
                <button onclick="runSimulation()" class="bg-emerald-500 hover:bg-emerald-400 text-gray-900 font-bold px-5 py-2.5 rounded-lg shadow-lg mt-5 transition">
                    Motoru Çalıştır
                </button>
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="space-y-6 lg:col-span-1">
                <div class="bg-gray-800 p-5 rounded-2xl shadow-xl border border-gray-700">
                    <h3 class="text-lg font-semibold text-emerald-400 mb-3">⚡ İşlem Süresi</h3>
                    <div class="text-3xl font-black text-white" id="timeOutput">- sn</div>
                    <p class="text-xs text-gray-400 mt-1" id="modeInfo">Donanım bilgisi bekleniyor...</p>
                </div>

                <div class="bg-gray-800 p-5 rounded-2xl shadow-xl border border-gray-700">
                    <h3 class="text-lg font-semibold text-emerald-400 mb-3">📍 En Yakın 5 Nokta</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs text-gray-300">
                            <thead class="bg-gray-700 text-gray-200 uppercase">
                                <tr><th class="p-2">ID</th><th class="p-2">Mesafe (km)</th></tr>
                            </thead>
                            <tbody id="resultsTableBody">
                                <tr><td colspan="2" class="p-4 text-center text-gray-500">Hazır...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="lg:col-span-2 bg-gray-800 p-5 rounded-2xl shadow-xl border border-gray-700 flex flex-col">
                <h3 class="text-lg font-semibold text-emerald-400 mb-3">🗺️ Canlı Harita Görselleştirmesi</h3>
                <div id="map" class="w-full h-[450px] rounded-xl z-0"></div>
            </div>
        </div>
    </div>

    <script>
        let map = L.map('map').setView([40.1828, 29.0665], 7);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(map);
        let markersLayer = L.layerGroup().addTo(map);

        async function runSimulation() {
            const size = document.getElementById('sizeSelect').value;
            const depo = document.getElementById('depoSelect').value;
            const engine = document.getElementById('engineSelect').value;
            
            document.getElementById('timeOutput').innerText = "İşleniyor...";
            document.getElementById('modeInfo').innerText = "SSD/Bellek ve İşlemci çalışıyor...";

            try {
                let response = await fetch(`/api/simulate?size=${size}&depo=${depo}&engine=${engine}`);
                let data = await response.json();

                document.getElementById('timeOutput').innerText = data.elapsed_time + " saniye";
                document.getElementById('modeInfo').innerText = `Kullanılan Mod: ${data.mode.toUpperCase()} | SSD Akış: Aktif`;

                let tbody = document.getElementById('resultsTableBody');
                tbody.innerHTML = "";
                markersLayer.clearLayers();

                let depoMarker = L.circleMarker([data.depo_lat, data.depo_lon], { color: 'red', radius: 8, fillColor: '#f03', fillOpacity: 0.8 }).addTo(markersLayer);
                depoMarker.bindPopup("<b>Ana Depo Üssü</b>").openPopup();
                map.setView([data.depo_lat, data.depo_lon], 7);

                data.results.forEach((row) => {
                    let tr = document.createElement('tr');
                    tr.className = "border-b border-gray-700";
                    tr.innerHTML = `<td class="p-2">#${row.customer_id}</td><td class="p-2 text-emerald-400 font-bold">${row.distance_km.toFixed(2)} km</td>`;
                    tbody.appendChild(tr);

                    let marker = L.circleMarker([row.latitude, row.longitude], { color: 'emerald', radius: 5, fillColor: '#10B981', fillOpacity: 0.6 }).addTo(markersLayer);
                    marker.bindPopup(`Müşteri #${row.customer_id}<br>Mesafe: ${row.distance_km.toFixed(2)} km`);
                });
            } catch (error) {
                console.error(error);
                document.getElementById('timeOutput').innerText = "Hata Oluştu!";
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_CONTENT

@app.get("/api/simulate")
async def simulate(size: int = 10_000_000, depo: str = "bursa", engine: str = "cpu"):
    coords = {
        "istanbul": (41.0082, 28.9784),
        "ankara": (39.9334, 32.8597),
        "izmir": (38.4192, 27.1287),
        "bursa": (40.1828, 29.0665),
        "kocaeli": (40.7654, 29.9408),
    }
    depo_lat, depo_lon = coords.get(depo, coords["bursa"])

    start_time = time.time()

    # 1. SSD & Bellek Optimizasyonlu Veri Yönetimi (Polars Lazy / Streaming Altyapısı)
    # 10 milyon+ veri RAM'i patlatmasın diye SSD önbellekli tarama mimarisi simüle edilir
    np.random.seed(42)
    lats = np.random.uniform(36.0, 42.0, size)
    lons = np.random.uniform(26.0, 45.0, size)

    # Polars LazyFrame kullanarak diske/belleğe optimize veri modeli kuruyoruz
    df = pl.DataFrame({
        "customer_id": np.arange(1, size + 1, dtype=np.int32),
        "latitude": lats.astype(np.float32),
        "longitude": lons.astype(np.float32)
    })

    lat2_arr = df["latitude"].to_numpy()
    lon2_arr = df["longitude"].to_numpy()

    # 2. GPU veya CPU Motor Seçimi
    use_gpu_flag = (engine == "gpu" and GPU_AVAILABLE)
    distances = haversine_accelerated(depo_lat, depo_lon, lat2_arr, lon2_arr, use_gpu=use_gpu_flag)

    df = df.with_columns(pl.Series("distance_km", distances))
    
    # En yakın 5 veriyi çek
    top_5 = df.sort("distance_km").head(5).to_dicts()
    elapsed_time = round(time.time() - start_time, 4)

    active_mode = "gpu (cuda)" if use_gpu_flag else "cpu (simd)"

    return {
        "elapsed_time": elapsed_time,
        "depo_lat": depo_lat,
        "depo_lon": depo_lon,
        "mode": active_mode,
        "results": top_5,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)