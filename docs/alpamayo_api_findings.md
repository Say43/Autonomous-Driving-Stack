# Alpamayo 1.5 — tatsächliche Inferenz-API (Code-Audit)

**Quelle:** `third_party/alpamayo1.5`, geklont von `https://github.com/NVlabs/alpamayo1.5`
(Klon erfolgreich, keine Umbenennung/Verschiebung).
**Commit:** `24179cfa8b2eeaf775e9e21698b23af0f899522d` ("Add bounded CUDA graph replay for diffusion expert (#36)", 2026-08-28)
**Umfang:** 41 Dateien, ~6.800 Zeilen. Keine Gewichte geladen, kein Modelllauf, keine Installation.

Alle Zeilennummern beziehen sich auf Dateien unterhalb von `third_party/alpamayo1.5/`.

---

## Konsequenzen für den Projektplan

Punkte, bei denen der Code dem Planungsdokument widerspricht oder es entscheidend präzisiert:

1. **Egomotion-Historie ist 1,6 s / 16 Wegpunkte — nicht 0,4 s.** Das Dokument vermischt zwei
   verschiedene Historien. Es gibt zwei unabhängige Historien-Streams:
   *Bilder* = 4 Frames pro Kamera über 0,3 s Spanne (`load_physical_aiavdataset.py:162-165`),
   *Egomotion* = 16 Wegpunkte über 1,5 s Spanne (`load_physical_aiavdataset.py:49,105-110`).
   Die 16 sind **hart erzwungen**: `helper.create_message` schreibt exakt 48
   `<|traj_history|>`-Platzhalter (`helper.py:106-109`), und der Delta-Tokenizer erzeugt
   3 Tokens pro Wegpunkt (`delta_tokenizer.py:72-80`) → 48/3 = 16. Eine abweichende Anzahl
   Wegpunkte bricht das `masked_scatter` in `replace_pad_token` (`base_model.py:89-93`).
   → **Der CARLA-Worker muss 16 Ego-Posen bei 10 Hz puffern, nicht 4.**

2. **Chain-of-Causation ist nicht abschaltbar.** Es gibt keinen Flag und keinen zweiten
   Codepfad. `sample_trajectories_from_data_with_vlm_rollout` führt *immer* zuerst einen
   autoregressiven VLM-Rollout aus (`alpamayo1_5.py:317-323`, bis zu `max_generation_length`
   Tokens, Default 256) und speist erst dessen KV-Cache in den Diffusions-Experten
   (`alpamayo1_5.py:330-331,364-372`). `return_extra=True` steuert nur, ob der Text
   *dekodiert und zurückgegeben* wird (`alpamayo1_5.py:424-432`) — nicht ob er erzeugt wird.
   → **Laufzeitbudget: bis zu 256 AR-Decode-Schritte pro Inferenz, plus 10 Diffusionsschritte.
   Keine Möglichkeit, den teuren Teil zu überspringen.**

3. **Die Kameranamen im Dokument existieren so nicht.** Der Code identifiziert Kameras über
   ganzzahlige Indizes 0–6 mit fester Semantik (`load_physical_aiavdataset.py:81-89`) und
   textuelle Anzeigenamen im Prompt (`helper.py:27-35`). Die 4 Default-Kameras sind
   `cross_left(0)`, `front_wide(1)`, `cross_right(2)`, `front_tele(6)` — aber die im Prompt
   eingebetteten Namen lauten "Front left camera", "Front camera", "Front right camera",
   "Front telephoto camera". Die Reihenfolge wird **nach Index aufsteigend sortiert**
   (`load_physical_aiavdataset.py:198-202`), also `[0, 1, 2, 6]`.
   → **Der Worker muss Index + Anzeigename exakt reproduzieren; die Sortierung ist Teil des
   Kontrakts, nicht die Aufrufreihenfolge.**

4. **Zeitstempel gehen gar nicht ins Modell.** Der Loader berechnet `relative_timestamps` und
   `absolute_timestamps` (`load_physical_aiavdataset.py:204-216`), aber keiner der Inferenzpfade
   liest sie: `model_inputs` enthält nur `tokenized_data`, `ego_history_xyz`, `ego_history_rot`
   (`test_inference.py:50-54`). Der Delta-Tokenizer verwirft `hist_tstamp`/`fut_tstamp` explizit
   (`delta_tokenizer.py:71`). Das Modell nimmt implizit ein festes `dt = 0.1 s` an
   (`unicycle_accel_curvature.py:47`).
   → **Zeitstempel-Felder im geplanten Worker-Interface sind tote Last. Stattdessen muss die
   Simulation exakt auf 10 Hz gesampelt werden — Jitter wird stillschweigend als 100 ms
   interpretiert.**

5. **Kein DeepSpeed, aber Flash-Attention-2 ist der Default und BF16 ist praktisch erzwungen.**
   `deepspeed` kommt im gesamten Repo nicht vor. Aber `attn_implementation` fällt ohne
   explizite Angabe auf `"flash_attention_2"` zurück (`base_model.py:225-227`); der
   dokumentierte Ausweg ist `attn_implementation="sdpa"` (README.md:176-182), unterstützt durch
   `_supports_sdpa = True` (`base_model.py:297`). Alle Beispiele laufen unter
   `torch.autocast("cuda", dtype=torch.bfloat16)` (`test_inference.py:59`), und der
   CUDA-Graph-Pfad hat BF16 hartkodiert (`diffusion_expert_cuda_graph.py:299`).
   → **Auf Turing (sm_75) ohne natives BF16 ist das der Hauptrisikopunkt. FP16 ist nirgends
   getestet und der Aktionsraum rechnet ohnehin in FP32 (`unicycle_accel_curvature.py:128`).
   Zusammen mit ~24 GB VRAM-Bedarf ist lokale Ausführung auf einer 6-GB-Karte ausgeschlossen —
   der Worker muss remote laufen.**

6. **Der Diffusionsteil ist billig und deterministisch steuerbar, der VLM-Teil nicht.**
   Flow Matching mit Euler, Default 10 Schritte (`flow_matching.py:35`). Seed wird ausschließlich
   über den globalen `torch.cuda.manual_seed_all(42)` gesetzt (`test_inference.py:58`) — es gibt
   keinen Seed-Parameter in der API. Der initiale Rauschzustand wird in
   `flow_matching.py:171` gezogen; die VLM-Sampling-Stochastik (`do_sample=True`,
   `alpamayo1_5.py:296`) hängt am selben globalen Generator.
   → **Determinismus für Golden-Tests ist nur über globalen Seed + fixe Batchgröße erreichbar,
   und CUDA-Graphs/Autocast können ihn brechen.**

---

## Schritt 2 — API aus dem Code

### 1. Modell laden

Es gibt **keinen eigenen Builder und keinen DeepSpeed-Zwang**. `Alpamayo1_5` erbt
`from_pretrained` unverändert von HuggingFace `PreTrainedModel` — im gesamten `src/` existiert
keine Überschreibung von `from_pretrained` (nur `from_pretrained_submodules`, `base_model.py:413`,
das für Training-from-scratch gedacht ist und die Basisgewichte einzeln lädt).

Kanonischer Aufruf (`test_inference.py:39`):

```python
model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16).to("cuda")
```

Ohne Flash-Attention (README.md:176-182):

```python
model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda")
```

Klassenhierarchie: `Alpamayo1_5(ReasoningVLA)` (`alpamayo1_5.py:87`),
`ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin)` (`base_model.py:292`).
Config-Objekt: `Alpamayo1_5Config(ReasoningVLAConfig)` (`config.py:23`), `model_type = "alpamayo1_5"`
(`config.py:26`), `ReasoningVLAConfig(PretrainedConfig)` (`base_model.py:204`).
Die Config wird beim `from_pretrained` aus dem Checkpoint gelesen; die im Code stehenden
Defaults (`base_model.py:209-247`) sind nur Fallbacks.

Submodule werden per Hydra aus der Config instanziiert (`alpamayo1_5.py:112-129`):
`action_space`, `diffusion`, `action_in_proj`, `action_out_proj`. Der Diffusions-Experte ist eine
zweite Kopie des Text-Stacks: `AutoModel.from_config(expert_config)` mit gelöschtem
`embed_tokens` (`alpamayo1_5.py:110-112`).

**DeepSpeed:** kein Vorkommen im Repo (`grep -rn deepspeed src/` → leer). Nicht erforderlich.

**Backbone laut Code:** `vlm_name_or_path` Default `"Qwen/Qwen3-VL-8B-Instruct"`
(`base_model.py:210`), instanziiert als `Qwen3VLForConditionalGeneration` (`base_model.py:386`).
"Cosmos-Reason" erscheint nur in einer README-Tabelle (README.md:204), nirgends im Code.

### 2. Input-Preprozessor

Der Preprozessor ist **nicht Alpamayo-eigen**, sondern der Qwen3-VL-Chat-Prozessor
(`helper.py:190-199`):

```python
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"   # helper.py:25
MIN_PIXELS = 163840                                  # helper.py:23
MAX_PIXELS = 196608                                  # helper.py:24
processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, min_pixels=..., max_pixels=...)
processor.tokenizer = model.tokenizer                # helper.py:198
```

Beachte: Prozessor von der **2B**-Variante, Tokenizer aber vom geladenen Modell — der Tokenizer
trägt die 768 Trajektorien-Tokens `<i0>…<i767>` und die Spezialtokens (`base_model.py:336-361`).

#### Bilder

| Aspekt | Befund | Beleg |
|---|---|---|
| Layout | `(N_cameras, num_frames, 3, H, W)`, also **CHW**, nicht HWC | `load_physical_aiavdataset.py:194,209` |
| Umwandlung | Rohdaten kommen als `(num_frames, H, W, 3) uint8` und werden mit `rearrange(..., "t h w c -> t c h w")` gedreht | `load_physical_aiavdataset.py:174-179` |
| Dtype / Wertebereich | `torch.uint8`, 0–255 (`torch.from_numpy` ohne Cast); Normalisierung macht der Qwen-Prozessor | `load_physical_aiavdataset.py:178` |
| Übergabe ans Modell | flachgezogen zu `(N_cameras*num_frames, 3, H, W)` via `.flatten(0, 1)` und pro Frame als `{"type": "image", "image": frame}` | `test_inference.py:36`, `helper.py:57,71` |
| Assertion | `frames.ndim == 4` mit Fehlertext `expected (N, C, H, W)` | `helper.py:104` |

→ **Das Dokument ("(4, H, W, 3) uint8 pro Kamera") ist im Layout falsch:** erwartet wird
`(4, 3, H, W)` uint8 pro Kamera, bzw. flach `(N_cam*4, 3, H, W)`.

**Auflösung:** Nirgends im Alpamayo-Code hartkodiert. Die effektive Größe folgt aus
`MIN_PIXELS`/`MAX_PIXELS` über den `smart_resize`-Algorithmus des Qwen-Bildprozessors
(transformers, Patch-Faktor 32). Nachgerechnet ergibt jede 16:9-Eingabe (1080×1920, 720×1280,
1088×1920) genau **320×576 = 184.320 Pixel = 180 Vision-Tokens pro Bild**. Das ist eine
*Ableitung*, kein Literal im Repo — 1080×1920 selbst ist im Code **nicht auffindbar**
(kommt aus dem Datensatz).

**Frame-Reihenfolge:** ältester zuerst, belegt durch
`load_physical_aiavdataset.py:161-165`:

```python
# Image timestamps: if num_frames=4, load at [t0-0.3s, t0-0.2s, t0-0.1s, t0]
image_timestamps = np.array(
    [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)], ...)
```

Für `i=0` ergibt sich `t0 - 0.3s`, für `i=3` genau `t0`. Zusätzlich nummeriert
`_build_image_content` die Frames pro Kamera aufsteigend als `"frame 0 "`, `"frame 1 "`, …
(`helper.py:63-73`), was die Reihenfolge im Prompt festschreibt.

**Kamera-Identifikation:** kein Enum, sondern zwei parallele Mechanismen.
Fester Index pro Kameraname (`load_physical_aiavdataset.py:81-89`):

```python
camera_name_to_index = {
    "camera_cross_left_120fov": 0,  "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2, "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}
```

und die Klartextnamen, die vor dem ersten Frame jeder Kamera in den Prompt geschrieben werden
(`helper.py:27-35`, verwendet in `helper.py:67-69`):
`0: "Front left camera", 1: "Front camera", 2: "Front right camera", 3: "Rear left camera",
4: "Rear camera", 5: "Rear right camera", 6: "Front telephoto camera"`.
Unbekannte Indizes fallen auf `f"Camera {cam_id}"` zurück (`helper.py:68`).
Die Reihenfolge ist nach Index sortiert (`load_physical_aiavdataset.py:198-202`), mit dem
Kommentar `# Sort by camera index to ensure consistent ordering [0, 1, 2, 6]`.
Die BEV-Layout-Tabelle in `viz_utils.py:181-187` bestätigt die Semantik
(0 = cross_left, 1 = front_wide, 2 = cross_right, 6 = front_tele).

#### Egomotion

| Key | Shape | Dtype | Beleg |
|---|---|---|---|
| `ego_history_xyz` | `(B=1, n_traj_group=1, 16, 3)` | `float32` | `load_physical_aiavdataset.py:61,147-149`; 4D-Assertion `base_model.py:107` |
| `ego_history_rot` | `(1, 1, 16, 3, 3)` | `float32` | `load_physical_aiavdataset.py:62,150-152` |

`n_traj_group` muss 1 sein: `assert n_traj_group == 1, "Only one trajectory group is supported
for inference."` (`alpamayo1_5.py:277`).

**Anzahl 16:** Default `num_history_steps: int = 16` mit Kommentar
`(default: 16 for 1.6s at 10Hz)` (`load_physical_aiavdataset.py:49`). Faktisch überspannen die
16 Punkte 1,5 s (`t0-1.5s … t0`, `load_physical_aiavdataset.py:103-110`). Hart erzwungen durch
die 48 Prompt-Platzhalter (`helper.py:106-109`) × 3 Tokens/Wegpunkt (`delta_tokenizer.py:72-80`).

**Koordinatensystem:** Ego-Frame zum Zeitpunkt t0, explizit dokumentiert
(`load_physical_aiavdataset.py:128-130`):

```python
# Transform to local frame (relative to t0 pose)
# The model expects trajectories in the ego frame at t0.
# Transformation: xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
```

Folglich ist `ego_history_xyz[..., -1, :] == (0,0,0)` und `ego_history_rot[..., -1, :, :] == I`.
Der Aktionsraum verlässt sich darauf: *"we assume the traj_history_xyz[..., -1, :] is the current
position and it is all zero"* (`unicycle_accel_curvature.py:245,269-270`).

**Achsenkonvention:** Rechtshändig, **x vorne, y links, z oben** (FLU). Ein expliziter
Kommentar dazu fehlt; die Konvention ergibt sich zwingend aus dem Code:
- Yaw wird als Rotation um z aus `atan2(R[1,0], R[0,0])` gelesen (`geometry/rotation.py:25-38`,
  Kommentar `# phi is rotation about z`).
- Die 2D-Rotationsmatrix ist die Standard-CCW-Matrix `[[cos, -sin], [sin, cos]]`
  (`geometry/rotation.py:118-125`), eingebettet als z-Achsen-Rotation mit `[0,0,1]`
  in der dritten Zeile/Spalte (`geometry/rotation.py:206-213`).
- Vorwärtsintegration: `x += v·cos(θ)·dt`, `y += v·sin(θ)·dt` bei Start `θ=0`
  (`unicycle_accel_curvature.py:349-373`) → bei `θ=0` bewegt sich das Fahrzeug entlang +x,
  positive Krümmung dreht nach +y (links).

**Einheiten:** Meter und Sekunden. Achsenbeschriftung `"x (m)"` / `"y (m)"`
(`viz_utils.py:166-167`, `notebooks/inference.ipynb` Zelle 11), `dt = 0.1`
(`unicycle_accel_curvature.py:47`), Beschleunigungsgrenzen `(-9.8, 9.8)` m/s²
(`unicycle_accel_curvature.py:47`), Krümmung `(-0.33, 0.33)` 1/m
(`unicycle_accel_curvature.py:48`). Der Delta-Tokenizer klemmt Schritt-Deltas auf
±4 m in x/y und ±10 m in z (`delta_tokenizer.py:26-27`) — bei 0,1 s entspricht das 40 m/s.

**Text/Prompt:** Drei-Rollen-Chat (`helper.py:124-142`):
- System: `"You are a driving assistant that generates safe and accurate actions."`
- User: Bildinhalte + `f"{hist_traj_placeholder}{route_section}{prompt_text}"`, wobei
  `hist_traj_placeholder = "<|traj_history_start|>" + "<|traj_history|>"*48 + "<|traj_history_end|>"`
  (`helper.py:106-109`), `route_section = f"<|route_start|>{nav_text}<|route_end|>"` nur bei
  Navigation (`helper.py:111-113`), und
  `prompt_text = "output the chain-of-thought reasoning of the driving process, then output the
  future trajectory."` (`helper.py:115-118`).
- Assistant (vorbefüllt): `"<|cot_start|>"` (`helper.py:140`).

Tokenisierung immer mit `add_generation_prompt=False, continue_final_message=True`
(`test_inference.py:42-49`) — der Assistant-Turn wird fortgesetzt, nicht neu begonnen.
Die 48 Platzhalter werden erst im Modell durch echte Trajektorien-Token-IDs ersetzt
(`fuse_traj_tokens` → `replace_pad_token` mit `masked_scatter`, `base_model.py:172-201, 89-93`).

**Zeitstempel:** werden **nicht** übergeben. Weder absolut noch relativ. Der Loader liefert sie
(`load_physical_aiavdataset.py:204-216`), aber `model_inputs` enthält nur drei Keys
(`test_inference.py:50-54`, identisch in allen vier Notebooks), und der Tokenizer löscht die
Zeitstempel-Argumente sofort (`delta_tokenizer.py:71,121`). Einheit im Loader intern:
Mikrosekunden (`t0_us`), relative Werte in Sekunden (`load_physical_aiavdataset.py:206`).

### 3. Inferenzaufruf

```python
model_inputs = {
    "tokenized_data": inputs,            # BatchFeature vom Prozessor
    "ego_history_xyz": ...,              # (1,1,16,3)
    "ego_history_rot": ...,              # (1,1,16,3,3)
}
model_inputs = helper.to_device(model_inputs, "cuda")

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs, top_p=0.98, temperature=0.6,
        num_traj_samples=1, max_generation_length=256, return_extra=True,
    )
```
(`test_inference.py:50-67`)

Signatur (`alpamayo1_5.py:244-255`):

```python
def sample_trajectories_from_data_with_vlm_rollout(
    self, data, top_p=0.98, top_k=None, temperature=0.6,
    num_traj_samples=6, num_traj_sets=1, diffusion_kwargs=None, *args, **kwargs)
```

`max_generation_length` und `return_extra` sind **nicht in der Signatur**, sondern werden aus
`**kwargs` gelesen (`alpamayo1_5.py:289-291`, `alpamayo1_5.py:424`). Default für
`max_generation_length` ist `config.tokens_per_future_traj` (= 64).

Rückgabe: `(pred_xyz, pred_rot)` bzw. `(pred_xyz, pred_rot, extra)` bei `return_extra=True`
(`alpamayo1_5.py:432,433`). Der Docstring behauptet fälschlich `logprob` als drittes Element
(`alpamayo1_5.py:271`) — tatsächlich ist es das Text-Dict.

`extra` ist ein Dict mit den Keys `"cot"`, `"meta_action"`, `"answer"`
(`token_utils.py:163`), jeweils `np.ndarray` der Form `[B, num_traj_sets, num_traj_samples]`
(`alpamayo1_5.py:426-430`).

**Steuerung der CoC-Erzeugung:** *Kein Flag, kein separater Aufruf, kein zweiter Forward-Pass.*
Die Methode ist strukturell zweistufig, aber beide Stufen sind zwingend:
1. `self.vlm.generate(...)` mit `do_sample=True`, `max_new_tokens=max_generation_length`,
   gestoppt beim ersten `<|traj_future_start|>` via `StopAfterEOS`
   (`alpamayo1_5.py:296-323`, `token_utils.py:171-206`). Ein `ExpertLogitsProcessor` verbietet
   dabei die diskreten Trajektorie-Tokens (`alpamayo1_5.py:53-84`, gesetzt `alpamayo1_5.py:307-314`).
2. Der resultierende KV-Cache wird als Prefix an den Diffusions-Experten weitergereicht
   (`alpamayo1_5.py:330-331`, `step_fn` in `alpamayo1_5.py:358-386`).

`return_extra` beeinflusst nur Schritt 4 (Dekodieren des Textes). Die Reasoning-Tokens werden
in jedem Fall generiert — das ist der dominante Laufzeitposten.

**Weitere Einstiegspunkte:**
- `generate_text(data, top_p=0.98, top_k=None, temperature=0.6, num_samples=1,
  max_generation_length=256) -> dict[str, np.ndarray]` (`base_model.py:456-465`) — reiner
  VQA-/Textpfad ohne Trajektorie, benötigt nur `data["tokenized_data"]`
  (`notebooks/inference_vqa.ipynb` Zelle 8/12).
- `sample_trajectories_from_data_with_vlm_rollout_cfg_nav(...)` (`alpamayo1_5.py:440`) — gleiche
  Signatur, zusätzlich Classifier-Free Guidance über zwei parallele KV-Caches.
- `nav_utils.compare_nav_conditions(model, processor, data, nav_text, ...)` (`nav_utils.py:~85`)
  — Komfortwrapper, der drei Varianten (mit Nav / ohne / gespiegelt) fährt.

### 4. Ausgabe

| Tensor | Shape | Dtype | Beleg |
|---|---|---|---|
| `pred_xyz` | `(B, num_traj_sets, num_traj_samples, 64, 3)` | wie `traj_history_xyz` (float32) | `alpamayo1_5.py:414-419`, `unicycle_accel_curvature.py:374-381` |
| `pred_rot` | `(B, num_traj_sets, num_traj_samples, 64, 3, 3)` | dito | `alpamayo1_5.py:417-419`, `unicycle_accel_curvature.py:387` |

Reshape-Beleg (`alpamayo1_5.py:414-416`):
`einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples)`.
Indexierung in den Beispielen: `pred_xyz[0, 0, i, :, :2]` (`notebooks/inference.ipynb` Zelle 11),
Docstring `[B, n_traj_group, K, T, 3]` (`viz_utils.py:34`).

**64 Wegpunkte / 6,4 s bei 10 Hz:** `n_waypoints: int = 64` (`unicycle_accel_curvature.py:51`),
`dt: float = 0.1` (`unicycle_accel_curvature.py:47`) → 6,4 s. Zusätzlich
`num_future_steps: int = 64` mit Kommentar `(default: 64 for 6.4s at 10Hz)`
(`load_physical_aiavdataset.py:50`).

**Frame:** Ego-Frame bei t0, gleiche Konvention wie der Input. Startzustand ist explizit
`x=0, y=0, θ=0` (`unicycle_accel_curvature.py:349,362-363`). Die z-Komponente wird **nicht
vorhergesagt**, sondern vom letzten Historienpunkt kopiert
(`unicycle_accel_curvature.py:384-385`):

```python
# Handle only_xy case for output
traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]
```

Da `ego_history_xyz[..., -1, :]` per Konstruktion Null ist, ist z in der Ausgabe faktisch
konstant 0. `pred_rot` sind reine Yaw-Matrizen (`rot_2d_to_3d(rotation_matrix_torch(θ))`,
`unicycle_accel_curvature.py:387`) — Roll und Pitch sind immer 0.

**Sampling-Parameter des Diffusionsdecoders** (`diffusion_kwargs`, durchgereicht nach
`FlowMatching.sample`, `alpamayo1_5.py:400-406`; Signatur `flow_matching.py:52-66`):

| Parameter | Default | Beleg |
|---|---|---|
| `int_method` | `"euler"` (einzige Option, sonst `ValueError`) | `flow_matching.py:34,111-112` |
| `inference_step` | `None` → `num_inference_steps = 10` | `flow_matching.py:35,92` |
| `temperature` | `1.0` (skaliert das Startrauschen) | `flow_matching.py:64,171` |
| `use_classifier_free_guidance` | `False` | `base.py:51`, `flow_matching.py:93-94` |
| `inference_guidance_weight` | `1.0` | `flow_matching.py:36,95-96` |

CFG erfordert `unguided_step_fn` und damit die `_cfg_nav`-Methode
(`flow_matching.py:97-98`); Notebook-Beispiel mit `inference_guidance_weight: 1.5` und
`temperature: 0.6` in `notebooks/inference_nav.ipynb` Zelle 14, mit der Warnung
"To run this inference with CFG, you will need 60GB+ GPU memory" (Zelle 13).

**Seed:** Es gibt **keinen Seed-Parameter**. Determinismus nur global:
`torch.cuda.manual_seed_all(42)` unmittelbar vor dem Aufruf (`test_inference.py:58`, sowie in
allen Notebooks). Betroffen sind sowohl `torch.randn` für das Startrauschen
(`flow_matching.py:171`) als auch das VLM-Sampling (`do_sample=True`, `alpamayo1_5.py:296`).

### 5. Flexible Kameraanzahl

**Ja, im Code belegt** — nichts ist auf 4 fixiert:
- `camera_features: list | None = None` als Parameter; die 4-Kamera-Liste ist nur der
  `None`-Default (`load_physical_aiavdataset.py:35,73-79`).
- Alle Shapes sind über `N_cameras` parametrisiert (`load_physical_aiavdataset.py:194-196`).
- `_build_image_content` iteriert über `camera_indices` beliebiger Länge und verwendet
  `repeat_interleave(num_frames_per_camera)` (`helper.py:59-73`).
- Der Prozessor tokenisiert eine variable Anzahl Bilder; die Sequenzlänge wächst entsprechend
  (`notebooks/inference_cam_num.ipynb` Zelle 6 gibt `seq length` pro Konfiguration aus).
- Explizite Ablation mit 1, 2 und 4 Kameras im Notebook
  (`notebooks/inference_cam_num.ipynb` Zelle 6: `"1 cam (front wide)"`,
  `"2 cam (front wide + front tele)"`, `"4 cam (left + front wide + right + front tele)"`),
  alle mit demselben Modellobjekt (Zelle 8).
- VQA-Notebook läuft mit einer einzigen Kamera (`notebooks/inference_vqa.ipynb` Zelle 6).

Die Zahl der *Frames pro Kamera* ist dagegen weniger frei: `num_frames_per_camera=4` ist Default
in `helper.py:41,80,150` und muss mit `num_frames` im Loader übereinstimmen, sonst zerfällt die
`repeat_interleave`-Zuordnung (`helper.py:59`).

### 6. Kleinste Einstiegsstelle für einen eigenen Worker

**`third_party/alpamayo1.5/src/alpamayo1_5/test_inference.py`** (81 Zeilen, `main()` in Zeile 29)
— vollständiger End-to-End-Pfad: Daten laden → Message bauen → Modell laden → tokenisieren →
inferieren → minADE rechnen. Das ist die kompakteste Vorlage.

Weitere Entry-Points:
- `notebooks/inference.ipynb` — dieselbe Sequenz interaktiv, plus Visualisierung.
- `notebooks/inference_cam_num.ipynb` — Vorlage für abweichende Kamerakonfigurationen.
- `notebooks/inference_nav.ipynb` — Navigationskonditionierung und CFG.
- `notebooks/inference_vqa.ipynb` — reiner Textpfad über `generate_text`.

Es gibt **kein** `demo`-Skript, keine CLI, keinen Server-Entry-Point. Kein `__main__` außer in
`test_inference.py:80-81`. `pyproject.toml` definiert keine `[project.scripts]`.

Für einen CARLA-Worker sind exakt drei Dinge nachzubauen, da `load_physical_aiavdataset`
zwingend `physical_ai_av` + HuggingFace-Streaming voraussetzt (`load_physical_aiavdataset.py:21,71`):
1. Bilder als `(N_cam, 4, 3, H, W)` uint8, kameraindex-sortiert.
2. `ego_history_xyz` `(1,1,16,3)` / `ego_history_rot` `(1,1,16,3,3)`, float32, im t0-Ego-Frame
   mit letztem Punkt = Ursprung.
3. `helper.create_message` + `processor.apply_chat_template` unverändert übernehmen.

### 7. Harte Laufzeitannahmen (Windows / Turing sm_75)

| Risiko | Befund | Fundstelle |
|---|---|---|
| **DeepSpeed** | **Nicht vorhanden.** Kein Import, keine Dependency. | `grep -rn deepspeed src/` leer; `pyproject.toml:5-20` |
| **FlashAttention-2** | Default, wenn `attn_implementation` nicht gesetzt wird: `if attn_implementation is None: attn_implementation = "flash_attention_2"`. Harte Dependency `flash-attn>=2.8.3`. | `base_model.py:225-227`; `pyproject.toml:17` |
| **SDPA-Fallback** | **Vorhanden und offiziell.** `_supports_sdpa = True`; Übergabe von `attn_implementation="sdpa"` an `from_pretrained` reicht. Der Diffusions-Experte **erzwingt** SDPA ohnehin selbst: *"The diffusion expert does not support FlashAttention 2. Force sdpa for the expert in that case."* | `base_model.py:297`; `alpamayo1_5.py:107-110`; README.md:167-183 |
| **BF16-Hartkodierung** | `model_dtype: str = "bfloat16"` als Config-Default; alle Beispiele unter `torch.autocast("cuda", dtype=torch.bfloat16)`; im CUDA-Graph-Pfad literal hartkodiert. Auf sm_75 gibt es kein natives BF16 → Emulation oder Fehler. | `base_model.py:218`; `test_inference.py:59`; `diffusion_expert_cuda_graph.py:299`; `tests/test_diffusion_expert_cuda_graph.py:35,85-86,115,131` |
| **CUDA fest verdrahtet** | `.to("cuda")` in allen Beispielen; `helper.to_device(..., "cuda")` in `nav_utils.py` fest verdrahtet (kein Device-Parameter). Kein CPU-Pfad vorgesehen. | `test_inference.py:39,56`; `nav_utils.py:150` (`return helper.to_device(model_inputs, "cuda")`) |
| **Autocast-Dekoratoren** | Der Aktionsraum deaktiviert Autocast explizit und rechnet in FP32 (`@torch.amp.autocast(device_type="cuda", enabled=False)`, 9 Stellen). Harmlos, aber `device_type="cuda"` ist fest. | `unicycle_accel_curvature.py:128,165,215,233`; `action_space/utils.py:77,161,237,316,402,488` |
| **CUDA-Graphs** | Optional, standardmäßig **aus** (`self._diffusion_expert_cuda_graph = None`). Nur nach explizitem `model.enable_diffusion_expert_cuda_graph(...)` aktiv, mit Eager-Fallback bei unbekannter Shape-Signatur. | `alpamayo1_5.py:138,141-153,393-398`; README.md:106-123 |
| **torch.compile** | Wird **nicht** aufgerufen. Nur defensive Spuren: ein Kommentar zu Fake-Tensor-Metadaten und ein `@torch._dynamo.disable()` auf `traj_to_action` (Trainingspfad). | `action_space/utils.py:303`; `unicycle_accel_curvature.py:232` |
| **Triton-Kernels** | Keine eigenen. Nur transitiv über torch/flash-attn. | keine Fundstelle in `src/` |
| **Linux-only Pfade** | Keine. Kein `os.uname`, kein `/tmp`, keine POSIX-Pfadliterale, kein `fork`. Die README-Installationsanweisung ist bash-zentriert (`curl … | sh`, `source a1_5_venv/bin/activate`), das ist aber Doku, nicht Code. | README.md:47-58 |
| **Python-Version** | `requires-python = "==3.12.*"` — **exakt gepinnt**. Lokal ist Python 3.13 installiert; `uv sync` würde die Umgebung ablehnen. | `pyproject.toml:4` |
| **torch-Version** | `torch==2.8.0` gepinnt, `transformers==4.57.1` gepinnt. Lokal ist torch 2.6.0+cu124 installiert. Der Code nutzt neue Cache-APIs (`CacheLayerMixin`, `Cache(...)`, `prompt_cache.crop(...)`), die in älteren transformers-Versionen fehlen. | `pyproject.toml:14,16`; `diffusion_expert_cuda_graph.py:20,68,299`; `alpamayo1_5.py:378` |
| **VRAM** | ~24 GB für `num_traj_samples=1`, ~40 GB für 16, ~60 GB mit CFG (H100-Messung). | README.md:33-41,244 |

### 8. Lizenzdateien

- **`LICENSE`** — vollständiger, unveränderter Apache-2.0-Text (202 Zeilen), Appendix ausgefüllt
  mit `Copyright 2026 NVIDIA` (LICENSE:190). Gilt für den **Inferenzcode**.
- **Datei-Header** — jede `.py`-Datei trägt `SPDX-License-Identifier: Apache-2.0` und
  `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES` (z. B. `config.py:1-14`).
- **README.md:255-258** — trennt explizit:
  > - **Inference code**: Apache License 2.0 – see [LICENSE](./LICENSE) for details.
  > - **Model weights**: OpenMDW-1.1 – see the HuggingFace Model Card for details.
- Es gibt **keine** Lizenzdatei für die Gewichte im Repo. **OpenMDW-1.1 ist nur als
  Verweis auf die Model Card genannt, der Volltext ist nicht auffindbar.** Für die
  Lizenz-Compliance-Prüfung muss die HF-Model-Card separat herangezogen werden.
- Weitere Governance-Dateien: `CONTRIBUTING.md`, `SECURITY.md` (verweist auf NVIDIAs VDP).
- Der Disclaimer (README.md:260-269) beschränkt die Nutzung ausdrücklich auf
  "research, experimentation, and evaluation" und stellt fest, dass es kein
  automotive-validierter Stack ist.

---

## Schritt 3 — Abgleich mit dem Planungsdokument

| # | Behauptung im Dokument | Status | Befund im Code |
|---|---|---|---|
| 1 | 4 Kameras: `front-wide`, `front-tele`, `cross-left`, `cross-right` | **ABWEICHEND** (Anzahl bestätigt, Benennung falsch) | Die 4 Default-Kameras stimmen inhaltlich (`load_physical_aiavdataset.py:73-79`), aber diese Namensform existiert nirgends. Der Code kennt Feature-Namen `camera_front_wide_120fov` etc. (`load_physical_aiavdataset.py:81-89`), ganzzahlige Indizes 0/1/2/6 und **davon abweichende Prompt-Klartextnamen** "Front left / Front / Front right / Front telephoto camera" (`helper.py:27-35`). Insgesamt 7 Indizes definiert. |
| 2a | 0,4 s Historie bei 10 Hz = 4 Frames pro Kamera | **ABWEICHEND** (4 Frames korrekt, Zeitspanne falsch) | 4 Frames bei 0,1 s Abstand spannen **0,3 s**, nicht 0,4 s: `[t0-0.3s, t0-0.2s, t0-0.1s, t0]` (`load_physical_aiavdataset.py:161-165`). `num_frames: int = 4` (`load_physical_aiavdataset.py:36`), `num_frames_per_camera: int = 4` (`helper.py:41,80`). |
| 2b | ältester zuerst | **BESTÄTIGT** | `load_physical_aiavdataset.py:161-165` (Kommentar + Formel), verstärkt durch aufsteigende `"frame {idx}"`-Beschriftung (`helper.py:63-73`). |
| 3a | Auflösung 1080×1920 (HxW) | **NICHT AUFFINDBAR** | Keine Auflösung im Alpamayo-Code hartkodiert. H und W kommen ungeprüft aus dem Datensatz (`load_physical_aiavdataset.py:174-179`). Die einzige Assertion prüft `ndim == 4` (`helper.py:104`). |
| 3b | Prozessor sampelt intern auf 320×576 | **BESTÄTIGT (abgeleitet)** | Nicht als Literal vorhanden, folgt aber zwingend aus `MIN_PIXELS = 163840` / `MAX_PIXELS = 196608` (`helper.py:23-24`) über Qwens `smart_resize` (Faktor 32): jede 16:9-Eingabe → 320×576 = 184.320 Pixel = 180 Vision-Tokens. Nachgerechnet für 1080×1920, 720×1280 und 1088×1920 → identisches Ergebnis. |
| 4a | Egomotion: 16 Wegpunkte bei 10 Hz | **BESTÄTIGT** (aber 1,5 s Spanne, nicht 0,4 s) | `num_history_steps: int = 16` mit Kommentar "16 for 1.6s at 10Hz" (`load_physical_aiavdataset.py:49`); faktische Spanne `t0-1.5s … t0` (`load_physical_aiavdataset.py:103-110`). Hart erzwungen: 48 Prompt-Platzhalter (`helper.py:106`) ÷ 3 Tokens pro Wegpunkt (`delta_tokenizer.py:72-80`) = 16. |
| 4b | (x,y,z) + 3×3-Rotationsmatrix | **BESTÄTIGT** | `ego_history_xyz` `(1,1,16,3)`, `ego_history_rot` `(1,1,16,3,3)` (`load_physical_aiavdataset.py:61-62,147-152`); Assertion auf 4D in `base_model.py:107`. |
| 5a | Output: 64 Wegpunkte über 6,4 s bei 10 Hz | **BESTÄTIGT** | `n_waypoints = 64`, `dt = 0.1` (`unicycle_accel_curvature.py:47,51`); `num_future_steps: int = 64` "64 for 6.4s at 10Hz" (`load_physical_aiavdataset.py:50`). |
| 5b | Position + Rotationsmatrix | **BESTÄTIGT, mit Einschränkung** | `(…,64,3)` und `(…,64,3,3)` (`unicycle_accel_curvature.py:374-387`). Aber: z wird nicht vorhergesagt, sondern kopiert (`unicycle_accel_curvature.py:384-385`), und die Rotation ist reine Yaw-Einbettung ohne Roll/Pitch (`rot_2d_to_3d`, `geometry/rotation.py:197-213`). |
| 5c | Ego-Frame | **BESTÄTIGT** | Ego-Frame bei t0, Startzustand `x=y=θ=0` (`unicycle_accel_curvature.py:349,362-363`); Eingangstransformation `xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)` (`load_physical_aiavdataset.py:128-137`). Achsen (x vorne, y links, z oben) sind aus `geometry/rotation.py:25-38,118-125,206-213` und der Integration (`unicycle_accel_curvature.py:364-373`) ableitbar, aber **nirgends explizit kommentiert**. |
| 6a | Intern: dynamische Aktionen (Beschleunigung, Krümmung) | **BESTÄTIGT** | `UnicycleAccelCurvatureActionSpace`, `accel, kappa = action[..., 0], action[..., 1]` (`unicycle_accel_curvature.py:326`); Aktionsraum-Dims `(n_waypoints, 2)` = `(64, 2)` (`unicycle_accel_curvature.py:100-102`); Grenzen ±9,8 m/s² und ±0,33 1/m (`unicycle_accel_curvature.py:47-48`). Werte sind normalisiert (mean/std, `unicycle_accel_curvature.py:328-333`). |
| 6b | Unicycle-Modell in BEV | **BESTÄTIGT** | Klassen-Docstring "Unicycle Kinematic Model with acceleration and curvature as control inputs" (`unicycle_accel_curvature.py:38`); Trapez-Integration nur in x/y mit z-Kopie (`unicycle_accel_curvature.py:342-385`). |
| 7a | Backbone Cosmos-Reason2 8.2B | **NICHT AUFFINDBAR / ABWEICHEND** | Der Code instanziiert `Qwen3VLForConditionalGeneration` mit Default `vlm_name_or_path = "Qwen/Qwen3-VL-8B-Instruct"` (`base_model.py:210,376-390`). "Cosmos-Reason" steht nur in einer README-Tabelle (README.md:204), "Cosmos-Reason2" und "8.2B" nirgends. Der Prozessor kommt von `Qwen/Qwen3-VL-2B-Instruct` (`helper.py:25`). Checkpoint-Name: `nvidia/Alpamayo-1.5-10B` (`test_inference.py:39`). |
| 7b | Diffusions-Action-Expert 2.3B | **NICHT AUFFINDBAR** | Keine Parameterzahl im Code. Der Experte ist eine Kopie der Text-Config mit Overrides aus `config.expert_cfg` (`alpamayo1_5.py:100-112`) — die tatsächliche Größe steht nur in der Checkpoint-Config. Das Diffusionsverfahren ist **Flow Matching mit Euler**, nicht klassische Diffusion (`flow_matching.py:22-34`). |
| 7c | BF16 | **BESTÄTIGT** | `model_dtype: str = "bfloat16"` (`base_model.py:218`), `dtype=torch.bfloat16` in allen Beispielen (`test_inference.py:39,59`), hartkodiert im CUDA-Graph-Pfad (`diffusion_expert_cuda_graph.py:299`). |
| 8 | Mindestens 1 GPU mit 24 GB VRAM | **BESTÄTIGT (nur Doku, nicht Code)** | README.md:33-41 (Tabelle: 24/40/60 GB) und README.md:244 ("at least 24 GB VRAM", getestet auf RTX 3090/A100/H100/B200). Im Code selbst gibt es **keine** VRAM-Prüfung. Gilt für `num_traj_samples=1`; 16 Samples brauchen ~40 GB, CFG ~60 GB. |

---

## Ergänzende Befunde ohne Entsprechung im Dokument

- **Zeitstempel gehen nicht ins Modell** (siehe Konsequenz 4). Das Modell hat keinerlei
  Information über tatsächliche Frame-Abstände; `dt = 0.1 s` ist implizit.
- **Historien-Trajektorie wird diskret tokenisiert**, nicht als kontinuierlicher Vektor
  eingespeist: 1000 Bins pro Achse, Delta-Kodierung, Clamp auf ±4 m (x,y) bzw. ±10 m (z)
  pro 0,1-s-Schritt (`delta_tokenizer.py:26-30,72-80`). Bei >40 m/s wird still geklemmt.
- **`num_traj_sets` ist ein Fallstrick:** Die generierte Sample-Anzahl ist
  `num_traj_samples * num_traj_sets` (`alpamayo1_5.py:275`), aber `generation_config.
  num_return_sequences = num_traj_samples` (`alpamayo1_5.py:299`) — nur `num_traj_samples`
  wird an den VLM weitergegeben. Für den Standardfall `num_traj_sets=1` konsistent.
- **`generation_config` wird global am Modell mutiert** (`alpamayo1_5.py:293-304`) — die
  Sampling-Parameter bleiben nach dem Aufruf gesetzt und lecken in nachfolgende
  `generate_text`-Aufrufe.
- **`replace_padding_after_eos` schreibt in-place** in `vlm_outputs.sequences`
  (`token_utils.py:245`).
- **`data` wird tief kopiert** (`copy.deepcopy(data)`, `alpamayo1_5.py:273`) — die Eingaben
  werden nicht mutiert, aber das kostet bei großen Bild-Tensoren Speicher und Zeit.
- **`tokenized_data["input_ids"]` wird per `pop` entfernt** (`alpamayo1_5.py:281`) — wegen des
  deepcopy betrifft das nur die interne Kopie.
- **Warnung ohne Abbruch:** Fehlt das `<|traj_future_start|>`-Token in der Generierung, wird nur
  geloggt und auf die letzte Position zurückgefallen (`alpamayo1_5.py:170-179`). Bei zu kleinem
  `max_generation_length` erzeugt das stillschweigend unsinnige Trajektorien.
- **Tests:** Nur `tests/test_diffusion_expert_cuda_graph.py`, und der benötigt eine CUDA-GPU
  mit BF16 (`tests/test_diffusion_expert_cuda_graph.py:35`). Keine CPU-lauffähigen Tests.
