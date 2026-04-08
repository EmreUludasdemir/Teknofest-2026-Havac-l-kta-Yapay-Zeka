# Havacılıkta Yapay Zeka Yarışması 2026 Derin Araştırma ve Depo Analizi Raporu

**Tarih:** 2026-04-07  
**Kaynak:** Deep research analysis from official sources + repository inspection

> Status note (2026-04-08): Some wire-format and Task 1 mapping gap claims below are stale relative to the current spec-compliance patch. See `reports/competition_deep_analysis_2026_gap_audit.md` for stale-claim history and current repo status.

---

## Yönetici Özeti

Bu rapor, yarışmanın resmi kaynaklarındaki gereksinimleri, kod deposunun mevcut durumunu ve örnek veri seti yapısını tek bir teknik çerçevede birleştirir.

### Ana Bulgular

| Görev | Ağırlık | Durum | Kritik Boşluk |
|-------|---------|-------|---------------|
| Görev 1 | %25 | Kısmen hazır | `motion_status` wire'da yok |
| Görev 2 | %40 | Hazır (frozen) | — |
| Görev 3 | %25 | KRİTİK | `detected_undefined_objects` wire'da YOK |
| Rapor+Sunum | %10 | — | — |

### Kritik Uyumsuzluklar

1. **Görev 3 wire format yoksunluğu:** `detected_undefined_objects` gönderilmiyor → **%25 puan riski**
2. **motion_status wire format yoksunluğu:** Taşıt hareket durumu gönderilmiyor → AP düşüşü
3. **Sınıf eşlemesi eksikliği:** UAP(2)/UAİ(3) ayrımı model seviyesinde doğrulanmalı

---

## Yarışma Teknik Gereksinimleri

### Görev Seti ve Sınıflar

| Sınıf ID | Sınıf Adı | Ek Durum |
|----------|-----------|----------|
| 0 | Taşıt | motion_status: 0=hareketsiz, 1=hareketli |
| 1 | İnsan | — |
| 2 | UAP Alanı | landing_status: 0=uygun değil, 1=uygun, -1=alan değil |
| 3 | UAİ Alanı | landing_status: 0=uygun değil, 1=uygun, -1=alan değil |

### Veri Akışı Parametreleri

- Oturum süresi: ~5 dakika
- FPS: 7.5
- Kare sayısı: ~2250
- Çözünürlük: Full HD veya 4K
- Format: jpg/png
- Akış: Tek tek kare (toplu indirme YOK)

### JSON Çıktı Formatı (Taslak)

```json
{
  "detected_objects": [
    {
      "cls": 0,
      "top_left_x": 100,
      "top_left_y": 100,
      "bottom_right_x": 200,
      "bottom_right_y": 200,
      "motion_status": 1,
      "landing_status": -1
    }
  ],
  "detected_translations": [
    {
      "translation_x": 1.5,
      "translation_y": 2.3,
      "translation_z": 50.0
    }
  ],
  "detected_undefined_objects": [
    {
      "object_id": "ref_1",
      "top_left_x": 300,
      "top_left_y": 300,
      "bottom_right_x": 400,
      "bottom_right_y": 400
    }
  ]
}
```

### Görev 2 Health Davranışı

- İlk 1 dakika (450 kare): health=1 garanti
- Sonra: health=0 olabilir (süre belirsiz)
- health=1: Referansı aynen göndermek serbest
- health=0: Kendi kestirimi zorunlu

---

## Depo Analizi: Boşluklar

### KRİTİK: Wire Format Eksiklikleri

| Eksik Alan | Etki | Çözüm |
|------------|------|-------|
| `detected_undefined_objects` | Görev 3 = 0 puan | Wire adaptöre ekle |
| `motion_status` | AP düşüşü | Wire adaptöre ekle |
| `object_id` (Task 3) | Eşleme puanlanamaz | Wire adaptöre ekle |

### Mevcut Mimari (İyi Durumda)

- `MvpFrameProcessor`: 3 görevi koordine ediyor ✓
- `Task2Estimator`: health-aware hibrit strateji ✓
- `ProtocolOrchestrator`: Tek kare/tek sonuç protokolü ✓
- Adaptör katmanı konfigüre edilebilir ✓

---

## Öncelikli Eylem Planı

### P0: Wire Format Uyumu (ACİL)

```
1. FinalSequentialAdapter.build_wire_prediction() güncelle:
   - motion_status ekle (cls=0 için)
   - detected_undefined_objects ekle (Task 3)
   - object_id ekle

2. OfficialRepoBatchAdapter.build_wire_prediction() güncelle:
   - Aynı alanları ekle

3. Test güncelle:
   - test_json_schema.py Task 3 alanlarını doğrulasın
```

### P1: Sınıf Eşlemesi

```
1. Task1Detector._canonical_class_from_model_name() güncelle:
   - UAP → 2
   - UAİ → 3
   - Eşleme tablosu konfigürasyona taşı

2. Model fine-tune sırasında 4 sınıf hedefle
```

### P2: Score-Up (Öncelik Sırası)

1. Task 3 YOLOE (en yüksek ROI)
2. Task 1 RF-DETR + SAHI
3. Task 2 DPVO/SLAM (sadece mimari değişiklikle)

---

## Model Önerileri

### Görev 1: Nesne Tespiti

| Aday | Doğruluk | Hız | VRAM |
|------|----------|-----|------|
| YOLO-n/s fine-tune | Orta→Yüksek | Çok hızlı | Düşük |
| RT-DETR-R50 | Yüksek | Orta | Orta→Yüksek |

**Motion Status:**
- Tracker + heuristik (hızlı, label flip riski)
- X3D-XS clip classifier (daha sağlam)

**Landing Status:**
- UAP/UAİ kutusu etrafında "boş mu?" sınıflandırıcı
- Opsiyonel SAM maske doğrulama

### Görev 2: Pozisyon Kestirimi

Mevcut `Task2Estimator` yaklaşımı uygun:
- PhaseCorr + LK + drift guard
- Health=1'de kalibrasyon
- Health=0'da görsel VO

### Görev 3: Görüntü Eşleme

| Aday | Doğruluk | Cross-modality |
|------|----------|----------------|
| ORB + homography (mevcut) | Düşük→Orta | Zayıf |
| SuperPoint + LightGlue | Yüksek | Daha iyi |

**Öneri:** LightGlue tabanlı eşleme + DINO/CLIP cross-modality backup

---

## Kaynak Bağlantıları

```
Teknik Şartname 2026:
https://cdn.teknofest.org/media/upload/userFormUpload/2026_HAVACILIKTA_YAPAY_ZEKA_TEKNIK_SARTNAME_TR_v1_2026_02_21_W48vr.pdf

LightGlue:
https://github.com/cvg/LightGlue

YOLOE (Ultralytics):
https://docs.ultralytics.com/models/yoloe/

SAM:
https://github.com/facebookresearch/segment-anything
```

---

## Sonuç

**Acil eylem:** Wire format eksikliklerini gidermeden yarışmaya katılmak %25+ puan kaybı demektir.

**Sıralama:**
1. Wire format uyumu (P0 - acil)
2. Task 3 YOLOE score-up (P1 - yüksek ROI)
3. Task 1 RF-DETR (P2 - production zaten güçlü)
4. Task 2 (P3 - sadece mimari değişiklikle)
