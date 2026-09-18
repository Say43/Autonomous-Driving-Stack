# Open-Loop: Alpamayo auf der Kreuzungsszene Town10HD / Seed 21

Lauf `results/scene_town10_seed21` (45 s, 25 Fahrzeuge, echtes Rig +0.7 m, JPEG), 20 Pakete
alle ~2.1 s, Batch-Worker Version 7 (Konfiguration wie M0 v23). Beleg:
`results/batch_worker_v7_scene_town10_seed21.json`, Plaene `results/batch_worker_v7_plans.json`.

## Reasoning vs. Szene

| t | Plan | Reasoning | Was das Ego real tat |
|---|---|---|---|
| 2.5 s | 62 m, 9.8 m/s | Keep distance to the lead vehicle | Anfahrt |
| 4.5-8.8 s | 8-9 m, ~1.4 m/s | **Stop** to keep distance ... since it is stopped ahead | Halt 7.2-16.2 s (Schlange an roter Ampel) |
| 10.8-12.9 s | 19 m, 2.9 m/s | Adapt speed to maintain a safe distance | steht |
| 15.1 s | 24 m, 3.7 m/s | **Resume speed** since the lead vehicle starts moving | faehrt bei 16.2 s an |
| 17-19 s | 66-72 m, 10-11 m/s | Keep distance | beschleunigt auf 8 m/s |
| 21.4 s | 32 m, y=+19.6 | Yield to the cut-in vehicle from the right | Kreuzung |
| 25-34 s | 50-64 m, y=+8..+33 | Keep distance | 90-Grad-Linkskurve |
| 36-42 s | 74 -> 38 m, y~0 | Keep distance | geradeaus, verzoegert |

Alle 20 Plaene endlich. Das Modell benennt jede Phase; den Halt schreibt es dem stehenden
Vorausfahrenden zu, nicht der Ampel (bei 25 Fahrzeugen bildet sich eine Schlange; das Auto
voraus ist, was es sieht). Kurvenrichtung stimmt mit dem Ego ueberein (links, y+).

## Open-Loop-Abweichung gegen den tatsaechlichen Ego-Pfad (CARLA-Autopilot)

Plan im Rig-Weltframe gegen die Ego-Positionen bei t+0.1..t+6.4 s.

| Situation | ADE |
|---|---|
| Halt (4.5-8.8 s) | 1.2-1.5 m |
| Kurveneinfahrt (25.6-27.6 s) | 0.6-1.3 m |
| Ende (40-42 s) | 0.2-0.7 m |
| freie Fahrt (2.5, 17, 19, 21 s) | 13-15 m |
| mitten in der Kurve (30-34 s) | 3.8-6.7 m |

Mittel 5.46 m, Median 3.48 m ueber 20 Plaene.

**Lesart.** Die Fehler sind fast ausschliesslich longitudinal: das Modell plant 10-11 m/s, der
Autopilot faehrt 8 m/s; bei 2.5 s plant es 62 m Fahrt, bevor die Schlange sichtbar ist. Wo das
Ego steht oder lenkt, liegt das Modell auf 0.6-1.5 m -- Spur, Anhalten und Kurvenrichtung sind
richtig. Die Referenz auf echten Daten (M0) liegt deutlich darunter (Zahl privat, Dataset-Lizenz).

**Vorbehalt.** Die "Wahrheit" hier ist der CARLA Traffic Manager, kein menschlicher Fahrer.
Gemessen wird Uneinigkeit mit CARLAs Fahrstil, nicht Fehler gegen menschliches Fahren. Ob
10-11 m/s auf dieser Strasse "falsch" sind, kann nur der NuRec-Vergleich (M6) entscheiden.
Ein zweiter Domain-Gap-Beitrag ist bekannt und unquantifiziert: das Rig sitzt 0.7 m weiter vorn
als im Erfassungsfahrzeug (Mesh-Anpassung, docs/m4_real_rig.md).
