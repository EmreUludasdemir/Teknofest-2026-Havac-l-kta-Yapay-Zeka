# 2026 Yarisma Raporu Gap Audit

**Tarih:** 2026-04-08  
**Kapsam:** `reports/competition_deep_analysis_2026.md` icindeki yuksek etkili iddialarin, mevcut lokal worktree gercegine gore siniflandirilmasi.

## Durum Etiketleri

- `still_true`: rapordaki iddia mevcut repo gercegiyle uyumlu.
- `stale`: rapordaki iddia mevcut repo gercegiyle celisiyor.
- `fixed_in_worktree`: lokal worktree'de cozum gorunuyor, ancak henuz commit/push edilmemis.
- `needs_validation`: iddia acikca yanlis degil, ama guncel kanitla yeniden dogrulanmali.
- `strategy_only`: bu madde repo gercegi degil, yol haritasi veya model onerisi niteliginde.

## Repo Truth Ayrimi

- **Published branch truth:** `main` branch HEAD commit'i `fe666fb`. Bu audit sirasinda push edilmis son gorunur durum budur.
- **Repo current truth:** ayni branch uzerinde lokal worktree'de uncommitted degisiklikler vardir.
- **Kaynak dogrusu:** bu audit, kullanicinin talebine uygun olarak lokal worktree'yi esas alir.
- **Onemli not:** `fixed_in_worktree` etiketi, bir problemin repo genelinde kalici olarak kapandigi anlamina gelmez. Yalnizca lokal worktree'de cozum bulundugunu soyler.

## Yonetici Ozeti

- Mevcut arastirma raporunun en kritik iki bosluk iddiasi, yani **Task 3 wire alanlarinin eksikligi** ve **`motion_status` wire eksikligi**, mevcut worktree icin artik acik degildir. Bu maddeler `fixed_in_worktree` olarak siniflandirilmalidir.
- **Task 1 class mapping eksikligi** de worktree'de kismen ele alinmistir; UAP/UAi alias'lari eklenmis ve eski `airplane -> 2` davranisi kaldirilmis gorunmektedir. Bu da `fixed_in_worktree` durumundadir.
- Buna karsin bu cozumler **heniz publish edilmis branch gercegi degildir**; `git status --short` cikisi, bu alanlarin uncommitted oldugunu gostermektedir.
- `reports/competition_deep_analysis_2026.md` dosyasi UTF-8 byte olarak kayitli gorunuyor, ancak mevcut shell/rendering ortaminda mojibake uretiyor; bu nedenle yayina hazir belge gibi ele alinmamali ve encoding/rendering hijyeni acikca not edilmelidir.
- Rapordaki model ve roadmap bolumlerinin buyuk kismi **repo gercegi degil**, stratejik oneridir; bu bolumler `strategy_only` olarak okunmalidir.

## Iddia-Durum Tablosu

| Rapordaki iddia | Mevcut repo gercegi | Durum | Kanit | Onerilen aksiyon |
| --- | --- | --- | --- | --- |
| `detected_undefined_objects` wire'da yok, bu nedenle Task 3 kritik eksik | Lokal worktree'de hem sequential hem batch adapter `detected_undefined_objects` alanini wire payload'a ekliyor | `fixed_in_worktree` | `src/server/final_sequential_adapter.py:220-247`, `src/server/official_repo_batch_adapter.py:223-251`, `tests/test_json_schema.py:13-86` | Hedefli test ile dogrula, sonra uygun branch/PR ile commit/push et |
| `motion_status` wire'da yok | Lokal worktree'de adapterlar `motion_status` alanini `obj.motion_status is not None` durumunda wire'a yaziyor | `fixed_in_worktree` | `src/server/final_sequential_adapter.py:231-233`, `src/server/official_repo_batch_adapter.py:235-237`, `tests/test_json_schema.py:79-86` | Aynı degisiklik seti icinde commit/push et; external rapordan bu acik boslugu kaldir |
| Task 1'de UAP(2)/UAi(3) class mapping eksik | `_canonical_class_from_model_name()` icinde UAP/UAi alias'lari eklenmis; eski `airplane` eslemesi artik `None` donuyor | `fixed_in_worktree` | `src/task1/detector.py:25-59` | Gercek model label adlariyla runtime dogrulamasi yap, sonra commit/push et |
| "Acil P0 wire format uyumu gerekli" | Bu ifade publish edilmis branch icin henuz dogru olabilir; fakat lokal worktree icin stale hale gelmistir | `stale` | `git status --short`, adapter diff'i, `tests/test_json_schema.py` | Raporu "open gap" yerine "fixed in local worktree, not yet published" seklinde guncelle |
| JSON format final degil, taslak niteliginde | Bu madde repo tarafindan degil, 2026 sartname baglamindan geliyor; lokal kod da adapter katmanini ayri tuttugu icin hala gecerli bir uyari | `still_true` | Raporun kendi teknik kaynak yorumu; adapter katmani ayrik | Yayin materyalinde bu uyariyi koru |
| Task 2 "hazir/frozen" | Bu audit sirasinda Task 2 icin yeni runtime kaniti uretilmedi; rapor ifadesi ancak eski freeze/dogrulama baglaminda okunabilir | `needs_validation` | Bu worktree degisiklik seti Task 2'yi kapsamiyor | Task 2 hazirlik iddiasini yayin belgesinde tarihli kanitla bagla ya da daha temkinli ifade et |
| ORB baseline zayif, LightGlue tercih edilmeli | Bu repo-gercegi degil; stratejik mimari oneridir | `strategy_only` | Raporun model onerileri bolumu | Yol haritasi belgesine tasinmali; gap audit'te acik repo eksigi gibi sunulmamalı |
| YOLOE / RF-DETR / DPVO gibi sonrasindaki score-up onerileri | Bunlar repo'nun bugunku durumu degil, stratejik R&D yonleridir | `strategy_only` | Raporun "Model Onerileri" ve "Oncelikli Eylem Plani" bolumleri | Ayrik roadmap/backlog maddeleri olarak ele alinmali |

## Kritik Duzeltmeler

### 1. Task 3 wire alanlari artik worktree'de var

- `FinalSequentialAdapter.build_wire_prediction()` artik `detected_undefined_objects` alanini bos liste olarak baslatip sonuc nesnelerini wire'a yaziyor.
- `OfficialRepoBatchAdapter.build_wire_prediction()` icin de ayni davranis eklenmis.
- `tests/test_json_schema.py` artik `detected_undefined_objects` alaninin wire payload icinde bulunmasini zorunlu kiliyor.

**Audit sonucu:** rapordaki "Task 3 wire YOK" ifadesi bugunku lokal worktree icin gecerli degil.

### 2. `motion_status` artik worktree'de var

- Her iki adapter da `obj.motion_status is not None` oldugunda `motion_status` alanini wire'a ekliyor.
- JSON schema testi bu davranisi tasit nesnesi icin dogruluyor.

**Audit sonucu:** rapordaki "motion_status wire'da yok" ifadesi bugunku lokal worktree icin gecerli degil.

### 3. Task 1 class mapping worktree'de genisletilmis

- `src/task1/detector.py` icindeki `_canonical_class_from_model_name()` fonksiyonu artik:
  - `person/insan/human/pedestrian -> 1`
  - cesitli vehicle alias'lari -> 0
  - `uap/... -> 2`
  - `uai/... -> 3`
  - `airplane/aeroplane -> None`
- Bu, rapordaki "UAP(2)/UAi(3) ayrimi dogrulanmali" maddesini tamamen kapatmaz; fakat worktree seviyesinde onemli bir dogrudan duzeltmedir.

**Audit sonucu:** bu madde "open critical gap" degil, "fixed_in_worktree, runtime validation still needed" olarak okunmali.

## Hala Acik veya Dogrulama Isteyen Maddeler

- **Uncommitted durum:** Audit aninda `git status --short` su lokal worktree degisikliklerini gosteriyor:
  - `A reports/competition_deep_analysis_2026.md`
  - `M src/server/final_sequential_adapter.py`
  - `M src/server/official_repo_batch_adapter.py`
  - `M src/task1/detector.py`
  - `M tests/test_json_schema.py`
  - `?? reports/competition_deep_analysis_2026_gap_audit.md`
- **Published branch truth ile fark:** Bu degisiklikler commit/push edilmedigi surece rapordaki stale maddeler external repo okuyucusu icin hala gecerli gorunebilir.
- **Schema taslakligi:** JSON alan isimleri ve final wire beklentileri 2026 sartname revizyonuna acik oldugu icin adapter katmani esnek tutulmali.
- **Stratejik model onerileri:** YOLOE, LightGlue, RF-DETR, DPVO gibi oneriler repo'nun "bugun sahip oldugu capability" degil, ileri tasarim notlaridir.

## Rapor Hijyeni

- `reports/competition_deep_analysis_2026.md` dosyasi byte duzeyinde UTF-8 gorunse de, mevcut PowerShell goruntulemesinde belirgin mojibake veriyor:
  - Baslik ve Turkce karakterler `HavacÄ±lÄ±kta`, `YÃ¶netici Ã–zeti` benzeri bozuk gorunuyor.
- Bu nedenle mevcut rapor:
  - yayin/sunum materyali olarak dogrudan kullanilmamali,
  - companion audit olmadan teknik durumun guncel ozetini vermiyor,
  - kritik tablo ve "acil eksik" listesinde artik stale hale gelmis maddeler tasiyor.

## Kisa Oncelik Siralamasi

1. Mevcut worktree degisikliklerini hedefli test ile dogrula ve uygun branch/PR altina al.
2. `reports/competition_deep_analysis_2026.md` icindeki stale kritik bulgulari yayin materyalinden cikar veya bu audit'e referans ver.
3. Acik kalan stratejik bosluklari, gap audit'ten ayri bir roadmap/backlog dokumanina tasi.

## Dogrulama Notu

- Bu audit icin kullanilmis hedefli test:
  - `python -m unittest tests.test_json_schema -v`
