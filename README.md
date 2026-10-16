# Ulaşım Hatları Optimizasyonu (`ptopt`)

Bir şehirde yeni otobüs / metro hatlarının **duraklarının nereden geçeceğini** ve mevcut ve yeni
hatların **saatlik sıklıklarını** optimize ederek iki nokta arası yolculuk süresini ve trafiği
azaltmaya çalışan açık veri tabanlı bir model. İlk hedef İstanbul; şehre özel her şey bir YAML
dosyasında durduğu için başka şehirlere de uygulanabilir.

## Nasıl çalışır (kısaca)

1. **Bölgeler**: şehir 1 km'lik hücrelere bölünür; her hücrede nüfus (WorldPop 2026, 100 m) ve
   iş/çekim gücü (OSM iş yerleri) toplanır.
2. **Gün içi talep**: gravity modeli ile ev→iş, iş→ev ve diğer yolculuk matrisleri; saatlik
   ağırlıklar İBB İstanbulkart verisinden. Sabah akış X(ev)→Y(iş), akşam ters yönde.
3. **Örnekleme**: seçilen saatte K adet (X, Y) çifti **akışla orantılı** çekilir: akışın çoğu
   X→Y ise örneklerin çoğu X→Y olur. M simülasyon yapılıp ortalaması alınır.
4. **Süre hesabı**: frekans tabanlı çok modlu ağ (yürüme + bekleme + araç içi süre; otobüsler o
   saatteki trafik hızıyla). Z eşiğinin altında kalma oranı, ortalama süre ve araba payı ölçülür.
5. **Optimizasyon**: en kötü hizmet alan koridorlarda aday otobüs / metro hatları üretilir;
   bütçe (ek otobüs, metro km) altında greedy + simulated annealing ile hatlar ve durak yerleri,
   karekök kuralı ile dönem bazında sıklıklar optimize edilir.

Ayrıntılar: [`docs/methodology.md`](docs/methodology.md).

## Veri kaynakları

| Veri | Kaynak |
|---|---|
| Nüfus | [WorldPop](https://hub.worldpop.org) R2025A, 100 m, constrained (2015–2030) |
| İş yerleri (çekim) | OpenStreetMap, [BBBike İstanbul extract](https://download.bbbike.org/osm/bbbike/Istanbul/) veya Overpass |
| Otobüs + Metrobüs | İBB Açık Veri — [İETT GTFS](https://data.ibb.gov.tr/dataset/iett-gtfs-verisi) |
| Metro, Marmaray, tramvay (güncel) | OpenStreetMap hat ilişkileri (`route=subway/tram/train/...`): 2024 sonrası açılan hatlar dahil |
| Vapur | İBB Açık Veri — [Toplu Taşıma GTFS](https://data.ibb.gov.tr/dataset/public-transport-gtfs-data) |
| İl sınırı | OpenStreetMap (admin_level=4 "İstanbul") |
| Trafik | İBB Açık Veri — [Saatlik Trafik Yoğunluk](https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set) (geohash bazında hız) |
| Saatlik talep profili | İBB Açık Veri — [Saatlik Toplu Ulaşım](https://data.ibb.gov.tr/dataset/hourly-public-transport-data-set) |
| Kalibrasyon | Google Routes API (trafikli araba + toplu taşıma süresi), TomTom Routing API (trafikli araba) — anahtar gerekir, bkz. [`docs/api_keys.md`](docs/api_keys.md) |

İBB GTFS dosyalarındaki bozukluklar (Excel'in bozduğu koordinatlar, çift kodlanmış Türkçe
karakterler, sadece ilk durakta saat) yükleyici tarafından otomatik onarılır.

Google / Yandex / Bing harita karoları kullanım şartları nedeniyle kazınmaz; aynı veri bu
servislerin resmi API'lerinden alınır (`ptopt calibrate`).

## Kurulum

```bash
pip install -e ".[dev]"
```

## Kullanım

```bash
# 1) verileri indir (data/raw altına, bir kere)
ptopt download --config configs/istanbul_pilot.yaml

# 2) bölgeleri ve ağı kur, harita çiz
ptopt build    --config configs/istanbul_pilot.yaml

# 3) mevcut durumu değerlendir (08:00 ve 18:00)
ptopt evaluate --config configs/istanbul_pilot.yaml --hours 8 18

# 4) yeni hatları ve sıklıkları optimize et
ptopt optimize --config configs/istanbul_pilot.yaml

# 5) modeli Google / TomTom süreleriyle kalibre et (API anahtarı gerekir)
ptopt calibrate --config configs/istanbul.yaml

# 6) bütçe–süre eğrisi: günlük ortalama (08–20, 2 saatte bir) 50 dakikaya inene kadar
#    yatırım ekle; hangi yatırımla hedefe ulaşıldığını raporla
ptopt frontier --config configs/istanbul.yaml

# 7) süre nereye gidiyor? (yürüme / bekleme / araç içi / aktarma)
ptopt explain --config configs/istanbul.yaml --scenario results/istanbul/frontier_50dk/target/scenario.pkl
```

`ptopt frontier` çıktıları `results/<şehir>/<frontier.out_dir>/` altında: `summary.md`, `frontier.png`
(yatırım–süre eğrisi), `frontier.csv` (her adım), `level_025` … `level_100` (bütçenin o payı
harcandığında harita, GeoJSON, tam değerlendirme).

Çıktılar `results/<şehir>/` altında:

* `map_optimised.html`: interaktif harita (OpenStreetMap altlığı, katman aç/kapa, yeni durakların
  üzerine gelince mevcut durak mı yeni durak mı olduğu). Tarayıcıda açın.
* `map_optimised.png` / `map_base.png`: statik harita (Esri World Street Map altlığı).
* `summary.md`, `optimisation.json`: metrikler, yeni hatlar, bütçe.
* `new_lines.geojson`, `network_optimised.geojson`, `zones.geojson`: QGIS / geojson.io için.

Haritaları optimizasyonu tekrarlamadan yeniden çizmek için `ptopt map --config ...`.

İndirme gerektirmeyen deneme: `ptopt optimize --config configs/synthetic.yaml`.

### Konfigürasyon

| dosya | alan |
|---|---|
| `configs/istanbul_pilot.yaml` | Üsküdar – Kadıköy – Ataşehir – Ümraniye pilot alanı |
| `configs/istanbul.yaml` | tüm İstanbul |
| `configs/synthetic.yaml` | sentetik test şehri |

Önemli parametreler (`ptopt/config.py` içindeki varsayılanlar):

* `evaluation.Z_min`: X→Y süresi için hedef eşik (dakika)
* `evaluation.K`, `evaluation.M`: simülasyon başına örnek sayısı, simülasyon sayısı
* `evaluation.weighting`: `proportional` (varsayılan) veya `tempered` (`alpha` ile, 0.5 ≈ geometrik ortalama)
* `evaluation.corridors`: ayrıca raporlanacak belirli X→Y çiftleri
* `optimization.budget`: `extra_buses`, `metro_km`

### Başka bir şehir

`configs/istanbul.yaml`'ı kopyalayın; `bbox`, `population.iso3`, GTFS adresleri (ör.
[Mobility Database](https://mobilitydatabase.org)) ve OSM extract adresini değiştirin. Trafik
verisi yoksa `traffic.source: profile` saatlik varsayılan hız profilini kullanır.

## Testler

```bash
pytest
```
