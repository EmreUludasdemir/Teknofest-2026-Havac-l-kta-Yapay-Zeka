# Sequential Compatibility Checklist

Bu belge mevcut production sequential adapter davranisinin son bilinen resmi repo/API sekliyle uyumunu takip etmek icindir.

## Referans Snapshot
- Son okunabilen resmi repo snapshot commit'i: `ce249fc8042ada788df438d2c0a1b76e28d6beaa`
- Bu snapshot'ta kanitlanmis endpointler batch-sekillidir: `auth/`, `frames/`, `translation/`, `prediction/`, `session/`.
- Sequential endpoint ailesi (`session/open`, `session/next`, `session/prediction`, `session/close`) production mock/profile varsayimidir; final sunucu yayini olmadan tam kanit sayilmaz.

## Kanitlanmis Davranislar
- Production runner yalniz sequential calisir.
- `open_session -> warm-up -> fetch_next_frame -> send_wire_prediction -> fetch_next_frame` sirasi korunur.
- Bir frame icin sonuc gonderilmeden sonraki frame istenmez.
- Varsayilan wire profili `official_current` olup yalniz `session_id`, `frame_id`, `frame`, `detected_objects`, `detected_translations` gonderilir.
- `detected_undefined_objects` canonical modelde tutulur ama varsayilan wire'a cikmaz.

## Dormant Switch Path
- Teknik sartname taslagindaki undefined-object alanlari gerekirse `sequential.wire_profile = "draft_with_undefined"` ile acilir.
- Bu degisiklikten sonra sequential validator ve mock smoke yeniden alinmalidir.

## Son Kontrol Listesi
- Son erisilebilir resmi repo/API sekli yeniden okundu mu?
- Endpoint adlari (`auth/open/next/prediction/close`) final sistemle uyumlu mu?
- `frame_id` ve `session_id` alanlari final wire ile uyumlu mu?
- Undefined object alanlari final endpoint tarafindan acikca isteniyor mu?
- Retry ve timeout davranisi gercek sunucu ile tekrar smoke edildi mi?
