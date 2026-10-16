# Harici trafik / rota API anahtarları

Model, gerçek trafik ve toplu taşıma sürelerine göre kalibre edilebilir:
`ptopt calibrate --config configs/istanbul.yaml`. Bunun için aşağıdaki API'lerden en az biri gerekir.

Anahtarlar **ortam değişkeni** olarak okunur, hiçbir dosyaya yazılmaz:

| Değişken | Servis | Ne için |
|---|---|---|
| `GOOGLE_MAPS_API_KEY` | Google Routes API | trafikli araba süresi + toplu taşıma süresi |
| `TOMTOM_API_KEY` | TomTom Routing API | trafikli araba süresi (tipik hafta içi trafiği) |

> Harita karolarını (Google / Yandex / Bing haritası ekran görüntüsü) kazımak bu
> servislerin kullanım şartlarına aykırıdır. Proje yalnızca resmi API'leri kullanır.

## Google Maps Platform (Routes API)

1. <https://console.cloud.google.com> → üstten **yeni proje** oluştur.
2. **Billing** (Faturalandırma) → bir ödeme hesabı bağla. Kart istenir; aylık ücretsiz kullanım
   kotası vardır, kalibrasyon için gereken birkaç bin istek genellikle bu kotada kalır.
   Güncel ücretler için Google Maps Platform fiyat sayfasına bakın.
3. **APIs & Services → Library** → "**Routes API**" ara → **Enable**.
   (Eski "Distance Matrix API" gerekmez; proje yeni Routes API'yi kullanır.)
4. **APIs & Services → Credentials → Create credentials → API key**.
5. Anahtarı kısıtla: **API restrictions → Restrict key → Routes API**. İsteğe bağlı olarak bütçe
   uyarısı (Billing → Budgets & alerts) koyun.

## TomTom

1. <https://developer.tomtom.com> → **Register** (ücretsiz hesap).
2. Giriş yapınca **Dashboard → Keys** altında varsayılan bir anahtar otomatik oluşur.
   Routing ve Traffic API'leri ücretsiz katmanda günlük kota ile kullanılabilir.

## Yandex (opsiyonel, henüz bağlı değil)

<https://developer.tech.yandex.com> üzerinden "Router API / Distance Matrix API" için anahtar alınır.
Bu API'ler ticari lisanslıdır; ücretsiz kullanım sınırlı olabilir veya başvuru gerektirebilir.
Anahtar alınırsa `YANDEX_ROUTER_API_KEY` adıyla eklenip `ptopt/io/external.py`'ye bir adaptör
eklenebilir.

## Bing Maps

Bing Maps for Enterprise kapatıldı; yerini **Azure Maps** aldı. Gerekirse Azure Maps Route Matrix
API'si için de adaptör eklenebilir.

## Anahtarları ortama ekleme (Claude Code bulut oturumu)

1. Oturum başlığındaki **bulut ortamı menüsü → Edit**.
2. **Network secrets** (eski uygulamalarda **API credentials**) ya da **environment variables**
   bölümüne yukarıdaki değişken adlarıyla ekleyin.
3. Değişkenleri **yeni bir oturum** görür.
4. Anahtarı asla sohbete veya repoya yapıştırmayın.

Yerelde: `export GOOGLE_MAPS_API_KEY=...` / `export TOMTOM_API_KEY=...`, sonra `ptopt calibrate`.

## Ne kadar istek yapılır?

Varsayılan ayarlar (`calibration` bölümü, `ptopt/config.py`):

- saat başına 300 yolculuk × 2 saat (08:00, 18:00)
- Google: araba + toplu taşıma ≈ 1.200 eleman (üst sınır `max_google_elements: 4000`)
- TomTom: ≈ 600 istek (üst sınır `max_tomtom_requests: 1500`)

Tüm cevaplar `data/raw/external/` altında önbelleklenir; kalibrasyon tekrar çalıştırıldığında
yeni istek yapılmaz.
