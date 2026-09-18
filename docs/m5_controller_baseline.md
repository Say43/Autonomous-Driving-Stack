# M5 Controller-Baseline-Evidenz

Status: **Controller-Baseline bestanden**, vollständiges M5 mit Alpamayo noch
nicht bestanden. Implementierungs-Commit:
`8eff15f26b4fafb9f11c42f28f9f7ac7e7abeb6d`.

Die maschinenlesbare Evidenz steht in `results/m5_controller_evidence.json`.
Die große Trace bleibt lokal unter
`results/m5_controller_60s_final/trace.jsonl`; ihr SHA-256-Hash ist in der
Evidenzdatei festgehalten.

## Umfang

Der Lauf schließt den CARLA-Regelkreis mit denselben Komponenten, die später
Alpamayo-Pläne ausführen sollen:

- Pure Pursuit für die Querführung und PID für die Längsführung;
- Replanning alle 0,5 s (2 Hz), dazwischen Tracking des weltfest verankerten
  letzten Plans;
- Plausibilitätsprüfung gegen Krümmungs-, Beschleunigungs-, Geschwindigkeits-
  und Schrittweitenlimits;
- kontrolliertes Bremsen bei fehlendem, veraltetem oder unsicherem Plan;
- Collision-Sensor, Trace-Aufzeichnung und vollständiger Actor-/World-Cleanup.

Da CARLA und Alpamayo-1.5-10B nicht gemeinsam in die lokale 6-GiB-GPU passen,
liefert ein deterministischer CARLA-Lane-Waypoint-Planer die Testtrajektorien.
Der Lauf prüft deshalb die Controller-Infrastruktur, nicht das Modell.

## Verifiziertes Ergebnis

- 1.200 lückenlose Ticks bei 0,05 s: 60,0 s Simulationszeit;
- 120 eindeutige Replans, entsprechend exakt 2 Hz;
- 0 unsichere Replans, 0 Kollisionen und 0 Offroad-Ticks;
- 294,280 m Wegstrecke, 4,907 m/s Mittel und 5,009 m/s Maximum bei 5 m/s Soll;
- maximal 0,604 m Abstand zur Spurmitte;
- maximale Plankrümmung 0,2646 1/m bei Grenze 0,33 1/m;
- maximale Planbeschleunigung 2,1123 m/s² bei Grenze 9,8 m/s²;
- alle Stellgrößen in ihren Wertebereichen, nie gleichzeitig Gas und Bremse;
- danach 0 Fahrzeuge, 0 Walker und 0 Sensoren; Synchronmodus wieder aus;
- 87 Tests bestanden, Ruff und `git diff --check` ohne Fehler.

Zwei vorangegangene 60-s-Diagnoseläufe zeigten zunächst reale
Integrationsprobleme: ungleichmäßige CARLA-Waypoint-Abstände lösten die
Beschleunigungsgrenze aus, danach erzeugte eine zu kurze Rückführung zur
Spurmitte eine Krümmungsspitze. Die finale Implementierung resampelt die
Kartenroute nach gemessener Bogenlänge und verwendet eine quintische,
12 m lange Einfädelung. Die Sicherheitsgrenzen wurden nicht gelockert.

## Reproduktion

CARLA auf Town01 starten und anschließend aus dem Repository ausführen:

```powershell
.\.venv-sim\Scripts\python.exe scripts\run_closed_loop_baseline.py `
  --out results\m5_controller_60s_final `
  --ticks 1200 --seed 1 --target-speed 5 --purge-actors
```

## Offener Schritt zum vollständigen M5

`m5_passed` bleibt bewusst `false`. Für vollständiges M5 muss der
CARLA-Waypoint-Planer durch echte Alpamayo-Ausgaben ersetzt und derselbe
60-s-Nachweis erneut erbracht werden. Derzeit blockiert davor bereits M0:
Kaggle-CLI-Läufe erreichen weder den gespeicherten `HF_TOKEN` noch zuverlässig
die GPU. Der nächste sinnvolle externe Schritt ist der interaktive
Presence-only-Test in der angemeldeten Kaggle-Notebook-Sitzung; identische
CLI-Retries liefern keine neue Information.
