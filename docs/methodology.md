# Yöntem (Methodology)

Bu belge modelin her adımını ve neden öyle seçildiğini anlatır.

## 1. Bölgeler (zones)

Çalışma alanı (bbox) metrik bir projeksiyonda (otomatik UTM; İstanbul için EPSG:32635)
kare hücrelere bölünür (varsayılan 1 km). Her hücreye:

* **Nüfus**: WorldPop R2025A 100 m *constrained* raster'ının (varsayılan 2026 tahmini) piksel
  değerlerinin toplamı. Bu, *gece nüfusu*dur, yani insanların yaşadığı yerdir.
* **İş / çekim gücü**: OSM POI'lerinin ağırlıklı sayımı (ofis 10, sanayi 15, dükkân 3,
  zanaat 3, eğitim/sağlık/kamu vb. 2) ve nüfusun %10'u (yerel hizmet işleri). Gerçek
  istihdam verisi (ör. SGK işyeri istatistikleri) olursa `jobs` doğrudan onunla değiştirilebilir.

Nüfusu veya işi eşiğin altında olan hücreler atılır.

## 2. Gün içi talep (OD matrisi)

Yolculuklar üç amaca ayrılır:

| amaç | üretim (O) | çekim (D) | model |
|---|---|---|---|
| Ev → İş | nüfus | iş | çift kısıtlı gravity |
| İş → Ev | — | — | Ev→İş matrisinin transpozu |
| Diğer | nüfus + iş | nüfus + iş | çift kısıtlı gravity (daha kısa mesafe) |

Gravity: `T_ij = a_i O_i · b_j D_j · exp(-d_ij / d0)`, `a`, `b` Furness iterasyonuyla
bulunur, böylece satır toplamları O'ya, sütun toplamları D'ye eşit olur. `d0` (İstanbul'da
10 km) ortalama yolculuk mesafesini belirler.

Saat `h` için matris:

```
T(h) = günlük_yolculuk × profil[h] × ( s_ev→iş(h)·M_ev→iş + s_iş→ev(h)·M_ev→işᵀ + s_diğer(h)·M_diğer )
```

* `profil[h]`: İBB saatlik İstanbulkart biniş verisinden (hafta içi) hesaplanır.
* `s_ev→iş(h)`, `s_iş→ev(h)`: sabah zirvesinde ağırlıkla ev→iş, akşam zirvesinde iş→ev.
  Bu sayede "nüfusun çoğu X'te oturup Y'de çalışıyor" yapısı sabah X→Y, akşam Y→X akışı üretir.

## 3. X–Y çiftlerinin örneklenmesi

Her simülasyonda K adet (X, Y) çifti `T(h)`'den örneklenir.

* **Orantılı (varsayılan)**: `P(X=i, Y=j) = T_ij(h) / ΣT`. Saat 08:00'de akışın %60'ı X→Y,
  %30'u Y→Z, %10'u Z→X ise örneklerin yaklaşık %60'ı X→Y olur. Örnek ortalaması, *yolcu
  ağırlıklı* ortalama yolculuk süresinin yansız tahminidir — optimize ettiğimiz şey tam olarak bu.
* **Ilımlı (tempered)**: `P ∝ T^α`. α = 0.5 önerilen "geometrik ortalama" fikrine karşılık gelir:
  baskın yön yine çoğunlukta ama zayıf akışlar daha iyi temsil edilir. Ortalama alınırken
  `w = T / T^α` ağırlığı kullanılır, böylece metrik yine yolcu ağırlıklı kalır.

Örnekleme iki aşamalıdır: önce `origins_per_sim` kadar başlangıç noktası, sonra her örnek için
varış noktası. Böylece hesap maliyeti K ile değil başlangıç noktası sayısıyla ölçeklenir.

M simülasyon farklı tohumlarla tekrarlanır; sonuçlar ortalama ± standart sapma olarak verilir.
Tüm senaryolar **aynı örneklerle** değerlendirilir (common random numbers), böylece senaryo
farkları örnekleme gürültüsü değildir.

## 4. Toplu taşıma ağı

Frekans tabanlı bir graf, her saat için ayrı kurulur:

* Yürüme erişimi / çıkışı: bölge merkezinden 800 m içindeki duraklar, kuş uçuşu × 1.3, 4.5 km/s.
* Aktarma: 250 m içindeki duraklar arası yürüme.
* Biniş: `min(1 / (2·Σf), bekleme_tavanı) + biniş cezası`. Σf, aynı **sonraki iki durağa** giden
  tüm hatların o saatteki toplam sefer sıklığıdır ("ortak hatlar": aynı caddeden geçen 5 hattan
  hangisi önce gelirse ona binilir; Spiess–Florian hiperyol mantığının yaklaşığı).
* Headway her hat için GTFS'teki o saatteki sefer sayısından (`60 / sefer`, 3 saatlik kayan
  ortalama) hesaplanır. İETT bir hattı çok sayıda varyanta böldüğü için (farklı ilk durak, küçük
  sapmalar) aynı hat ve yöndeki varyantlar, durakları en az %70 ortak ise ana güzergâhta
  birleştirilip sıklıkları toplanır.
* Raylı ağ (metro, tramvay, Marmaray, banliyö, füniküler) OpenStreetMap hat ilişkilerinden kurulur
  (İBB'nin raylı GTFS'i 2024 tarihli ve yeni hatları içermiyor). OSM'de sefer saati olmadığı için
  zirve/zirve dışı headway mod bazında varsayılan değerlerden (`transit_osm.headways`) gelir.
* Araç içi süre:
  * GTFS tüm duraklarda saat veriyorsa (raylı sistem, vapur) doğrudan o süreler,
  * otobüs: `mesafe × 1.3 / min(trafik_hızı(konum, saat) × 0.75, 40 km/s) + 20 sn`,
  * metrobüs / metro / tramvay: moda göre sabit hız.
* Sonuç: `min(toplu taşıma, sadece yürüme)`; 180 dakikada erişilemeyen yolculuklar 180 dk sayılır.

Araba süresi: başlangıç–varış doğrusu üzerindeki trafik hızlarının harmonik ortalaması ile
`mesafe × 1.3 / hız + 5 dk` (park/erişim).

### Kalibrasyon (Google Routes + TomTom)

Talep ağırlıklı 600 örnek yolculuk (08:00 ve 18:00) için Google Routes API'den trafikli araba ve
toplu taşıma süreleri, TomTom'dan trafikli araba süresi alınır (hafta içi bir gün, o saatte kalkış).

* Araba: her yolculuk için modelin hızını gözleme eşitleyen çarpan hesaplanır; çarpanlar en yakın
  10 yolculukla mekânsal olarak yumuşatılıp İBB hız alanının üzerine uygulanır. Otobüs hızları da
  bu düzeltilmiş alandan türetilir.
* Toplu taşıma: otobüs hız katsayısı, biniş cezası, bekleme tavanı, yürüme dolambaç katsayısı,
  erişim yarıçapı ve raylı hız ölçeği, model süresi ile Google süresi arasındaki log farkın karesi
  en küçük olacak şekilde (Nelder–Mead) ayarlanır.

Sonuç `data/processed/calibration_<şehir>.json`'a yazılır ve sonraki tüm komutlarda otomatik uygulanır;
öncesi/sonrası hata istatistikleri `results/<şehir>/calibration.json`'dadır.

## 5. Metrikler

* ortalama kapıdan kapıya süre,
* `P(t ≤ Z)`: Z dakikanın altında kalan yolculuk payı,
* ortalama Z aşımı `E[max(0, t − Z)]`,
* **araba payı**: arabası olanlar için ikili logit
  `P(toplu taşıma) = 1 / (1 + exp(−(β (t_araba − t_tt) + ASC)))`, toplam araba payı
  `araç_sahipliği × (1 − P)`. Trafik azaltımının göstergesidir.
* Amaç fonksiyonu: `J = ortalama süre + w · ortalama Z aşımı`.

Değerlendirme 08:00'den 20:00'e kadar **her iki saatte bir** (8, 10, 12, 14, 16, 18, 20) yapılır.
Her saatin talep yönü (sabah ev→iş, akşam iş→ev), trafik hızları ve sefer sıklıkları o saate
aittir. **Günlük ortalama**, her saatin değeri o saatte yola çıkan yolcu sayısıyla (İstanbulkart
saatlik profili) ağırlıklandırılarak hesaplanır; yoğun 08:00 ve 18:00, sakin 12:00'den daha çok
sayılır. Optimizasyonun amaç fonksiyonu bu günlük ortalamadır.

## 6. Yeni hat adayları

1. Baz senaryoda zirve saatlerde her örnek yolculuğun **açığı** hesaplanır:
   `max(0, t − Z) + 0.5 · max(0, t_tt − t_araba)`.
2. Açıklar 3 km'lik kaba hücre çiftlerine (koridorlar) toplanır; en kötü koridorlar seçilir
   (çok kısa ve birbirine çok yakın olanlar elenir).
3. Koridor için bölge grid'i üzerinde en kısa yol bulunur; kenar maliyeti
   `uzunluk × (1 − ödül × yoğunluk)` olduğundan yol yoğun (nüfus + iş) bölgelerden geçmeye
   yönelir. **Hattın hangi bölgelerden geçeceği** böyle belirlenir.
4. **Otobüs hatları gerçek sokaklara oturtulur**: OSM'den (pbf) sürülebilir yol ağı (tek yönler
   dahil) kurulur. Koridor boyunca en kısa yol, kenar maliyeti
   `uzunluk × yol_sınıfı_katsayısı × (1 + (koridora_uzaklık / 750 m)²)` ile bulunur:
   ana arterler (primary 1.0, secondary 1.1, tertiary 1.3) mahalle sokaklarına (residential 2.5)
   tercih edilir ve hat koridordan uzaklaşmaz. Duraklar bu sokak geometrisi boyunca ~400 m
   aralıkla yerleştirilir; mevcut İETT durağına 150 m'den yakınsa o durak kullanılır. Durak arası
   süre gerçek yol mesafesiyle hesaplanır.
5. Metro: koridor sadeleştirilmiş düz bir hat olarak alınır, istasyonlar ~1.2 km aralıkla.

## 7. Optimizasyon

1. **Açgözlü (greedy)**: bütçe (ek otobüs sayısı, metro km) izin verdikçe amaç fonksiyonunu en
   çok düşüren adayı ekle.
2. **Benzetimli tavlama (simulated annealing)**: durak kaydır, hattı uzat/kısalt, headway değiştir,
   hat değiştir/ekle/çıkar. Bu adım durakların tam yerini ince ayarlar. Otobüs durakları sokak
   boyunca kaydırılır, uzatma da sokak ağı üzerinden rotalanır; hat yoldan çıkmaz.
3. **Sıklık (frekans) dağıtımı**, her dönem için (sabah zirvesi, öğle, akşam zirvesi, akşam):
   atama ile hat yükleri (biniş) bulunur, sonra sabit filo altında karekök kuralı uygulanır:
   `araç_l ∝ √(yolcu_l × tur_süresi_l)` (toplam bekleme süresini minimize eden çözüm).
   Değişiklik ancak o dönemin amaç fonksiyonunu iyileştiriyorsa kabul edilir.

Otobüs filosu ihtiyacı: `⌈tur_süresi / headway⌉`, tur süresi zirve saatteki trafik hızıyla
hesaplanan gidiş-dönüş süresi + %10 bekleme.

## 8. Bütçe–süre eğrisi (`ptopt frontier`)

Bütçe bugünkü ağa göre tanımlanır (ek otobüs: zirve filonun bir payı, yeni raylı km: raylı ağ
uzunluğunun bir payı). İki kullanım vardır:

* **Sabit bütçe** (örn. filonun %20'si, raylı ağın %30'u): bu bütçeyle süre nereye iner?
* **Hedef arama** (`target_min`, `stop_at_target`): bütçe üst sınırı geniş tutulur (filonun
  %100'ü, raylı ağın %200'ü) ve günlük ortalama hedefe (50 dk) indiği anda durulur. Sonuç,
  hedef için gereken yatırımı gösterir; hedefe ulaşma tam örneklemle doğrulanır.

Her adımda dört tür yatırım aday olur:

1. yeni metro hattı (koridor üzerinde, 1.2 km istasyon aralığı),
2. mevcut metro / raylı hattın bir ucundan 4 veya 8 km uzatılması (aktarma olmadan devam eder;
   yön, ileri 120°'lik koni içinde en yoğun nüfus+iş bölgelerine göre seçilir),
3. yeni otobüs hattı (sokaklar üzerinde),
4. en yoğun otobüs hatlarına ek araç (headway × 0.5, yani araç sayısı iki katı).

Seçim, normalleştirilmiş maliyet başına amaç fonksiyonu kazancına göre yapılır
(`maliyet = ek_otobüs / otobüs_bütçesi + km / km_bütçesi`). Her aday önce küçük bir örneklemle
(bir kaç temsilci saatte) taranır; en iyiler tüm saatlerde daha büyük örneklemle yeniden
değerlendirilir. Her adımda farklı hat / koridorlara ait birkaç iyi yatırım birlikte alınabilir
(birlikte daha iyiyse). Ağ büyüdükçe yeni hat adayları güncel ağın açıklarından yeniden üretilir.
Belirli bütçe kullanım seviyelerinde ve hedefe ulaşıldığında tam değerlendirme yapılır ve harita
çizilir.

## 9. Bilinen sınırlamalar

* Araç kapasitesi / kalabalık modellenmiyor (yükler raporlanıyor). Sonraki adım: kapasite
  kısıtlı atama.
* Gerçek OD verisi yok, gravity ile tahmin ediliyor. İBB İstanbulkart biniş–iniş (OD) verisi
  veya mobil telefon verisi elde edilirse `Demand` sınıfı doğrudan onunla değiştirilebilir.
* Araba süresi ve mevcut İETT hatlarının durak arası mesafesi (GTFS'te shape yok) hâlâ
  düz çizgi × 1.3 ile yaklaşıklanıyor; yeni otobüs hatları gerçek yol mesafesi kullanıyor.
* İBB raylı GTFS verisi 2024'e ait; yeni açılan hatlar (ör. M12, M14) eksik olabilir.
* Google / Bing trafik haritaları kullanım şartları gereği kazınmıyor; İBB'nin resmi saatlik
  trafik yoğunluğu verisi kullanılıyor. Başka şehirler için TomTom / HERE gibi lisanslı API'lere
  `TrafficField` arayüzü üzerinden adaptör yazılabilir.
