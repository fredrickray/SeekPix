# Face / search fixtures

Used for regression tests and threshold calibration. Never commit large
personal photo dumps — keep this folder to a handful of labeled images.

## Layout

```text
tests/fixtures/
├── README.md
├── pairs.example.json   # copy to pairs.json and edit
├── same_person/         # optional: drop images here for your own sorting
└── different/           # optional
```

## `pairs.json` format

```json
{
  "same_person": [
    ["same_person/alice_a.jpg", "same_person/alice_b.jpg"]
  ],
  "different_people": [
    ["different/bob.jpg", "different/carol.jpg"]
  ]
}
```

Paths are relative to `tests/fixtures/`.

## Calibrate

```bash
# From the SeekPix repo root, with .venv active:
python scripts/calibrate_face_threshold.py --pairs tests/fixtures/pairs.json
```

The script prints same-person and different-people score ranges and suggests a
threshold that sits between them (with a small margin).

Without labeled pairs you can still inspect scores against the live library:

```bash
python scripts/calibrate_face_threshold.py \
  --probe /path/to/face.jpg \
  --against-library
```
