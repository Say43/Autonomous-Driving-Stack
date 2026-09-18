# Simulationsumgebung (M4-Vorbereitung)

## Warum ein zweites Python-Environment

`carla==0.9.16` liefert Wheels für cp310, cp311 und cp312 — **nicht für cp313**.
Das Haupt-Environment (`.venv`, Python 3.13) kann CARLA also nicht installieren.
Gewählt wurde Python 3.12, weil auch der Alpamayo-Upstream `requires-python == 3.12.*`
verlangt; damit bleibt später ein gemeinsames Env möglich.

```
uv python install 3.12
uv venv --python 3.12 .venv-sim
VIRTUAL_ENV=.venv-sim uv pip install "carla==0.9.16" numpy pytest
```

## Verifizierte API-Oberfläche

Gegen die installierte Version geprüft (nicht aus Dokumentation übernommen):

| Aufruf | Status |
|---|---|
| `TrafficManager.set_random_device_seed` | vorhanden |
| `TrafficManager.set_synchronous_mode` | vorhanden |
| `Map.get_topology`, `Map.generate_waypoints` | vorhanden |
| `World.get_actors`, `World.tick`, `World.apply_settings` | vorhanden |
| `Sensor.listen`, `Actor.destroy`, `Vehicle.set_autopilot` | vorhanden |
| `WorldSettings.fixed_delta_seconds`, `.synchronous_mode`, `.no_rendering_mode` | vorhanden |

**Wichtiger Befund für die Koordinatentransformation:**
`carla.Transform.get_matrix()` liefert eine 4x4-Matrix, `get_inverse_matrix()` die
Inverse. Die Umrechnung von CARLAs linkshändigen (pitch, yaw, roll)-Grad auf
Rotationsmatrizen muss also nicht selbst implementiert werden. Das entschärft den
als häufigste Fehlerquelle bezeichneten Punkt erheblich — die Roundtrip-Tests aus
M1 bleiben trotzdem Pflicht, weil die Händigkeit und Achsenzuordnung gegenüber der
Alpamayo-Konvention weiterhin zu klären ist.

## Simulator-Paket

- Quelle: Release Notes von `carla-simulator/carla` Tag `0.9.16`
- `https://downloads.carlasim.com/Windows/CARLA_0.9.16.zip` — 7,28 GB
- `https://downloads.carlasim.com/Windows/AdditionalMaps_0.9.16.zip` — 6,75 GB

Nur das Basispaket wird geladen (Ziel: `../carla-0.9.16/`). Die Zusatzkarten
(Town06/07/10HD) werden auf M7 verschoben, wo die Szenarienvielfalt gebraucht wird;
für M4 genügen Town01–05. Grund: begrenzter Plattenplatz.

## Gemessen auf der Zielhardware (GTX 1660 Ti Max-Q, 6144 MiB)

CARLA 0.9.16, Town10HD_Opt, `-RenderOffScreen -quality-level=Low`, synchroner Modus
mit `fixed_delta_seconds=0.05`, Ego plus vier RGB-Kameras à 1920x1080 mit
`sensor_tick=0.1` (Alpamayo-Rig: front_wide 120°, front_tele 30°, cross_left/right 120°).

| Größe | Wert |
|---|---|
| VRAM mit vollem 4-Kamera-Rig | ~5700 MiB von 6144 MiB (~93 %) |
| VRAM nach Serverstopp | 0 MiB (Last war praktisch vollständig CARLA) |
| Durchsatz | 8,15 ticks/s = **0,41x Echtzeit** |
| Leere Welt ohne Sensoren | 14,3 ticks/s |
| Karten im Basispaket | Town01–05, Town10HD (je auch `_Opt`); Town06/07 fehlen |

**Bewertung.** Das 4-Kamera-Rig läuft, aber mit ~7 % Reserve. Ein zusätzlicher
VRAM-Verbraucher auf derselben GPU ist ausgeschlossen — das bestätigt die
Entscheidung, Alpamayo nicht lokal zu betreiben, unabhängig von der Modellgröße.
0,41x Echtzeit ist unkritisch, weil der Projektauftrag ausdrücklich keine
Echtzeitanforderung stellt.

Fällt die Reserve in dichteren Szenen mit Verkehr weg, ist der nächste Hebel die
2-Kamera-Konfiguration, die der Upstream-Code laut `docs/alpamayo_api_findings.md`
nachweislich unterstützt.

## Offener Punkt für M2/M4: Sensor-Frames sind nicht garantiert im Gleichtakt

Im Messlauf lieferte `front_wide` 21 Callbacks, während die drei anderen Kameras
20 lieferten (40 Ticks, `sensor_tick=0.1`). Ob das echter Sensor-Jitter ist oder ein
Wettlauf beim Zurücksetzen der Zähler im Messskript, ist damit **nicht entschieden**.

Für den Adapter ist die Ursache aber zweitrangig: Er darf Kamerabilder niemals über
die Reihenfolge oder Anzahl der Callbacks zuordnen, sondern ausschließlich über
`image.frame` des jeweiligen Sensordatums. Genau darauf zielt Fallstrick 3 des
Projektauftrags, und die 10-Hz-Prüfung in `SensorPacket` fängt einen Fehler hier ab,
statt ihn stumm ins Modell zu lassen.
