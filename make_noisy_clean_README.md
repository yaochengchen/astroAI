# How to generate `noisy/` and `clean/` for astroAI cleaner

This helper script creates paired FITS maps from DL3 event FITS:

- `noisy/` = integrated counts map from the EVENTS table
- `clean/` = `counts - background`, clipped at 0

## Run

```bash
python make_noisy_clean_from_dl3.py \
  -f path/to/your_conf.yml \
  -o path/to/output_pairs
```

This creates:

```text
output_pairs/
  noisy/
    crab_00001.fits
  clean/
    crab_00001.fits
  manifest.csv
```

Then point `preprocess.directory` at `output_pairs` and run:

```bash
python preprocess_ds.py -f path/to/your_conf.yml
```
