# M5 – Alpamayo schließt die Schleife (Zeitlupe)

Stand 2026-09-16 (abends). Ergänzt `docs/m5_controller_baseline.md` (Controller ohne Modell) um den
Pfad, in dem Alpamayo tatsächlich das Fahrzeug steuert.

Nachtrag 18.09.2026: Controller/Schutzschicht und Viewer wurden weiter gehärtet.
Die folgenden Modellläufe bleiben unverändert historische Ergebnisse.
Neue Implementierung, lokale Regressionstests und verbleibende Grenzen stehen in
[`quality_hardening_2026-09.md`](quality_hardening_2026-09.md).

## Idee

CARLA läuft synchron und **tickt nicht, solange ein Plan unterwegs ist**. Pro Replan:

```
CARLA (steht)  --4 Kameras x 4 Frames (F-Theta), 16 Ego-Posen-->  Job im HF-Repo
                                                                     |
Kaggle 2x T4: Alpamayo-1.5 NF4 (M0-v23-Konfiguration), ~11 s        v
CARLA: Controller faehrt den Plan --replan-interval Sim-Sekunden  <--  Plan im HF-Repo
```

Alpamayo sieht damit die Folgen seines eigenen vorigen Plans – eine echte geschlossene
Schleife, nur nicht in Echtzeit. Die Wanduhr-Latenz (30–60 s pro Schritt) wird nie zu
Planalter, weil die Simulationszeit dazwischen stillsteht. `summary.json` trägt deshalb
`alpamayo_used: true, closed_loop: true, real_time: false` und den gemessenen
`slowdown_factor`.

## Bausteine

| Teil | Datei | Aufgabe |
|---|---|---|
| Wire-Format | `src/acarla/loop/codec.py` | Job = die Ausgabe von `sensor_packet_to_model_inputs` mit JPEG-kodierten Frames (`--codec png` = bit-exakt wie die Batch-Tensoren, ~8x größer); Plan = `PlanResult` + Herkunft als JSON |
| Warteschlange | `src/acarla/loop/transport.py` | privates HF-Dataset-Repo `<hf-nutzer>/acarla-loop` (HF-Nutzer `say43`, nicht der Kaggle-Name): `runs/<id>/jobs/NNNNN.npz`, `runs/<id>/plans/NNNNN.json`, `runs/<id>/DONE`, `worker/heartbeat.json`, `worker/STOP`. `dir:<pfad>` = gleiche Schnittstelle auf einem Verzeichnis (Tests, Smoke) |
| Lokale Seite | `scripts/run_closed_loop_alpamayo.py` | Rig-Anbringung wie `run_open_loop.py`, Paketbau wie `build_model_inputs.py`, Controller wie `run_closed_loop_baseline.py`; wartet beliebig lange auf den Plan und zeigt dabei den Worker-Heartbeat |
| Worker | `notebooks/kaggle_loop_worker.ipynb` (Kernel `says43/alpamayo-loop-worker`) | lädt das Modell einmal, bedient dann Jobs; Heartbeat jede Minute; beendet sich nach 25 min ohne Job (`LOOP_IDLE_EXIT_MIN`) oder bei `worker/STOP` |
| Fake-Worker | `scripts/fake_loop_worker.py` | antwortet mit Geradeausplan; nur für den Test der CARLA-Seite |

Der Worker ist aus dem verifizierten M0-v23-Runner gebaut (Builder: Session-Scratchpad
`build_loop_worker.py`); Env, Quantisierung, `device_map` und Modell-Load sind byte-identisch
zum Batch-Worker. Die `acarla`-Module `types`, `loop.codec`, `loop.transport` werden als
Mini-Paket in den Kernel geschrieben, damit beide Seiten denselben Codec ausführen.

## Warum HF als Kanal

Der lokale Rechner hat keine öffentliche Adresse; der Kaggle-Kernel hat ausgehendes Internet
und das HF-Secret, das er ohnehin für die gated Backbone-Gewichte braucht. Jede Übertragung ist
ein kleiner Commit; beide Seiten pollen (Worker alle 3 s, lokal alle 3 s). Der Token wird von
keinem Skript angefasst: `huggingface_hub` liest ihn aus dem lokalen Login bzw. dem Kaggle-Secret.

## Sitzungen überleben

- Der Job liegt im Repo, bis ein Plan daneben liegt. Stirbt der Kaggle-Kernel (12-h-Limit,
  Quote, Absturz), wartet die lokale Seite – CARLA steht – und meldet jede Minute den
  Heartbeat-Zustand (`STALE` nach 4 min). Nach *Save & Run All* im Browser lädt der neue Kernel
  das Modell (~10–15 min) und nimmt den offenen Job auf; die lokale Seite merkt nichts außer der
  Wartezeit. `worker_sessions` in `summary.json` zählt, wie viele Kernel-Sitzungen beteiligt waren.
- Idle-Abschaltung schützt die 30 GPU-h/Woche: 25 min ohne Job → Kernel endet
  (`heartbeat.state = exited, exit_reason = idle_timeout`).
- Fehler im Worker werden als Plan mit `finite: false` beantwortet; lokal bremst der Controller
  (Plan veraltet), der nächste Intervall fragt neu, nach drei Fehlern in Folge bricht der Lauf ab.
- `--mark-others-done` markiert liegengebliebene Läufe als DONE, damit der Worker nur den
  aktuellen bedient. `--stop-worker-at-end` schreibt `worker/STOP`.

## Ablauf

1. **Einmalig lokal**: `hf auth login` in `.venv-sim` mit einem Token, der schreiben darf
   (das Repo wird beim ersten Lauf privat angelegt).
2. **Kaggle**: `says43/alpamayo-loop-worker` im Browser öffnen, Secret `huggingface` anhängen,
   *Save & Run All*. (Ein API-Push verliert die Secret-Bindung – nach jedem Push wiederholen.)
3. **CARLA** mit der Zielkarte starten, dann

   ```
   .venv-sim\Scripts\python scripts\run_closed_loop_alpamayo.py --out results\loop_town10_seed21 ^
       --seed 21 --map Town10HD_Opt --ticks 1200 --replan-interval 1.0 --traffic 20 --mark-others-done
   ```

   Rechnung: 60 s Simulation bei 1 s Replan = 60 Anfragen × (≈5 s Paketbau + Upload + ≈11 s
   Inferenz + Polling) ≈ 30–60 min. Die Reihenfolge 2 → 3 ist egal; Jobs warten.

4. Anschauen: `results/<lauf>/trace.jsonl` in den Viewer ziehen (Pläne sind schon im Trace),
   `plans.json` enthält Reasoning, Sicherheits-Gate-Werte und Wanduhrzeiten je Schritt.

## Betriebsgrenzen, die sich erst im Betrieb zeigten

- **HF-Commit-Limit: 128 Commits pro Stunde und Repo.** Ein Schritt = Job-Commit + Plan-Commit;
  Lauf 1 starb nach 47 Schritten am 429. Seitdem Jobs auf `<base>-j0/-j1`, Pläne auf
  `<base>-p0/-p1` (Seq mod 2), Heartbeats nur im Leerlauf alle 5 min, 429 wird bis 70 min
  abgewartet. Damit sind 1-s-Replans (~130 Schritte/h) knapp unter dem Limit je Shard.
- **Prozessstart:** Ein aus einer Tool-Shell gestarteter Python-Prozess stirbt mit der Shell.
  Der Lauf wird deshalb über den Windows-Taskplaner gestartet (`C:\Users\frede\acarla_loop.bat`,
  Task `acarla_loop`), Ausgabe nach `results/loop_town10_seed21.{log,err}`.
- **HF-Nutzer ≠ Kaggle-Nutzer** (`say43` vs. `says43`): Namespace wird zur Laufzeit aus `whoami`
  bestimmt.
- **Worker-Code-Refresh:** Der Kernel lädt `code/acarla_pkg.json` aus dem Basis-Repo (von der
  lokalen Seite bei jedem Start veröffentlicht). Codec-/Transport-Änderungen brauchen keinen
  Kernel-Push mehr, nur einen Kernel-Neustart – und der braucht das Secret ohnehin.

## Läufe am 2026-09-16 (Town10HD_Opt, Seed 21, 15 Fremdfahrzeuge, Replan 1 s)

| Lauf | Controller | Ergebnis | Befund |
|---|---|---|---|
| 1 `…_partial49s` | Geschwindigkeit zum Planalter | 47 Pläne, 49 s, 0 Kollisionen, 0 unsichere Pläne; Abbruch durch 429 | **Auto stand 15 s** trotz 35-m-Plänen: Alpamayo rollt aus dem Stand sanft an (0,7 m/s nach 1 s), der Tracker sah bei 1-s-Replans nur die Nullphase. Danach: Rot erkannt → Halt → „turns green" → Anfahren → Linkskurve mit 5 m/s sauber. Reasoning sagt „turn right", Geometrie ist links (y+, verifiziert). |
| 4 `…_run4_collision14s` | Fortschritt entlang des Plans, 2 s Vorausschau | Anfahren funktioniert (7,7 m/s nach 2 s); **Kollision bei 14,4 s mit 0,3 m/s** | Ego kriecht in einen stehenden Lincoln, während jeder Plan „Stop to keep distance" sagt, aber 6–9 m Weg enthält: **das Modell überschätzt den Abstand zum stehenden Vorausfahrenden** auf CARLA-Bildern. |
| 5 `…_run5_aeb` | wie 4 + Ground-Truth-AEB | 60 s komplett, 58 Pläne, 115 AEB-Ticks, **Kollision bei 35,6 s mit 4,9 m/s** (Palme, Gehweg), 697 Off-Road-Ticks | AEB fängt das Auffahren ab (t = 8–16 s, Standoff 1,5 m). Ab t = 21 s Drift nach rechts (Spurfehler 0,3 → 2,6 m bei Lenkwinkel ≈ 0); bei 10 m/s plant das Modell eine weite Linkskurve (80 m in 6,4 s), der Controller holt auf **15 m/s** auf, Spurfehler 4–5 m, Bordstein. |
| 6 `…_run6_speedbound` | wie 5 + Ziel ≤ Plan-Eigengeschwindigkeit + 1 m/s, Limit 10 m/s | **60 s komplett, 316 m, Ø 5,3 m/s, max 9,85 m/s**, 58 Pläne, 50 AEB-Ticks, 3 Kollisions-Ticks (Streifen bei 46,4 s, 3,9 m/s), 403 Off-Road-Ticks | Dieselbe Linkskurve wie in Lauf 5 diesmal sauber. Bei 43–46 s blockiert ein **geparkter Lincoln (statisches Objekt, kein Actor → AEB blind)** die Spur; das Modell schreibt „Nudge to the left“, plant aber y = −6 m (rechts, Gehweg) → Streifen, danach befreit es sich selbst und fährt bis 60 s weiter (Spurwechsel, „yellow traffic light“, Linkskurve). Achsenkonvention numerisch nachgeprüft (CARLA-Linkskurve → Historie y+, FLU y+ → CARLA links, Kameras per Bild): Kette konsistent, der Widerspruch liegt zwischen Text und Trajektorie des Modells – wie in Lauf 1. |

Was die Läufe zeigen, unabhängig von Kollisionen: Das Reasoning reagiert auf die *eigene* Fahrt
(„Stop for the red traffic light" → „Accelerate … turns green"; „Wait for a gap to merge right"
→ „Lane change to the right"). Ampeln werden erkannt (rot/grün genannt, in Lauf 1 an der
richtigen Stelle). Alle 58 + 58 + 47 + 13 Pläne bestanden das Dynamik-Gate (Krümmung, Beschleunigung).
Was sie auch zeigen: Der Abstand zu stehenden Fahrzeugen wird überschätzt, und ohne
Geschwindigkeitsbegrenzung fährt die Kombination Modell + Controller zu schnell in Kurven.
Ob Modell oder Controller „schuld" ist, trennt der Vergleich Lauf 4 (Modell ohne Netz) gegen
Lauf 5/6 (mit Netz): Die AEB-Eingriffe sind der Preis, den das Modell auf CARLA zahlt.

## Verifiziert / offen

- Verifiziert: Transportkette lokal ↔ Kaggle über HF (Round-Trip 25–30 s, davon ~11 s Inferenz,
  ~5 s Paketbau, ~4 s Upload), Sitzungswechsel des Workers ohne Eingriff lokal, Idle-Exit,
  Fake-Worker-Smoke, 152 Unit-Tests.
- Offen: Text-vs-Trajektorie-Widersprüche des Modells (2×), statische Hindernisse unsichtbar für die AEB (Actor-basiert), Abstandsüberschätzung (Rig-Verschiebung +0,7 m als Kandidat: die Kamera sitzt 0,7 m
  weiter vorn als am Referenzfahrzeug), Rechtsdrift auf Geraden, 2-s-Replan als Alternative
  (halbiert Commits und Laufzeit), Ground Truth ist der Traffic Manager.
