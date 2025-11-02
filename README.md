# Turkai v0.2 — Hızlı Başlangıç

## 1) Kur
```bash
cd Turkai_v0_2
cp .env.example .env
docker compose up -d ollama
docker exec -it turkai-ollama bash -lc "ollama pull mistral:7b-instruct"
docker compose up -d api
```

## 2) Belgeleri içeri al
```bash
# Klasöre dosya kopyala ve indeksle
# (ornek.txt zaten var)
curl -X POST http://localhost:8011/ingest
```

## 3) Sohbet
```bash
curl -sS http://localhost:8011/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Turkai nedir? Kısaca anlatır mısın?"}]}'
```

- Yanıt JSON'u her zaman `{"answer": "...", "session_id": "..."}` şeklinde gelir.
- Aynı oturumu sürdürmek için bir sonraki çağrıda `session_id` alanını tekrar gönderin:

```bash
SESSION=$(curl -sS http://localhost:8011/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"You are a precise coding assistant."},{"role":"user","content":"Merhaba"}]}' \
  | jq -r '.session_id')

curl -sS http://localhost:8011/chat \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION\",\"messages\":[{\"role\":\"user\",\"content\":\"Devam\"}]}"
```

## 4) Tarayıcı arayüzü
`http://localhost:8011/ui` adresinde basit bir web arayüzü vardır. İlk mesajda sistem promptunu yazıp sohbeti başlatın; oturumlar otomatik sürdürülür, "New chat" ile sıfırlanır. "Close" butonu, mevcut oturumu kilitleyip ekranda `turkai resume <id>` ipucunu gösterir.

## 5) Oturum özeti (resume)
- CLI üzerinden son konuşmaları listelemek:
  ```bash
  turkai resume
  ```
- Belirli bir oturumu incelemek:
  ```bash
  turkai resume <SESSION_ID> --show
  ```
- Aynı oturuma devam etmek:
  ```bash
  turkai resume <SESSION_ID> "Yeni komut"
  ```
- API ile manuel kullanım:
  ```bash
  curl -sS http://localhost:8011/resume | jq
  curl -sS "http://localhost:8011/resume?session_id=$SESSION" | jq
  ```

## 6) Kuantum araç demoları (LLM tetikli)
Araç çağırma açık olduğunda (`ALLOW_TOOLS=1`), kuantum görevleri için
LLM aşağıdaki gibi bir `toolcall` üretebilir (otomatik). Bunu tetiklemek için
soruya örneğin şöyle yazın:

> "4 düğümlü çember grafının MaxCut çözümünü hesapla ve sonucu açıkla."

İşlem sonunda JSON içinde `tool_result` alanını ve özet cevabı göreceksiniz.

## 7) Fluxion entegrasyon stub
`FLUXION_BIN` içinde bir çalıştırılabilir varsa (veya konteyner içine kopyalarsanız),
`tool: "fluxion.run"` ile sınırlı bayraklarla kısa bir koşum yapılabilir.
Bu yalnızca **izinli/lab ortamları** içindir.
