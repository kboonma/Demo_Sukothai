# Sukhothai Drought Watch (prototype)

Tambon-level drought early-warning prototype for Sukhothai province, December 2026 outlook.
Static site for GitHub Pages: `index.html` + JSON files in `data/`.

> **Status:** concept prototype. Index values are **synthetic (mock)** until the Earth Engine exports are processed; `meta.json` → `"mock": true` shows a banner on the page. The DDPM drought-declaration layer is real open data.

## Structure

```
index.html                      dashboard (Leaflet + Chart.js from cdnjs, no build step)
data/
  tambons.geojson               86 tambons + latest values, December outlook, alert levels
  timeseries.json               monthly indices per tambon, 2018 → latest
  meta.json                     data month, mock flag, alert counts
  tambon_alerts.csv             priority list for DDPM / district offices
  source/                       inputs: simplified boundaries, DDPM records (FY2562–2568)
scripts/
  sukhothai_drought_gee.ipynb   Colab notebook: Earth Engine → monthly tambon CSVs in Google Drive
  sukhothai_tambon_gee.geojson  boundaries to upload when running the notebook
  compute_indices.py            CSVs → SPI/SPEI/VHI/soil-moisture/water anomalies → indices → data/
raw/                            (git-ignored) put the six Earth Engine CSVs here
```

## Update the data

1. Run `scripts/sukhothai_drought_gee.ipynb` in Google Colab (sign in with your own Google account; needs an Earth Engine-registered Cloud project).
2. Download the six CSVs from Drive folder `sukhothai_drought/` into `raw/`.
3. `python scripts/compute_indices.py --raw raw/ --boundaries data/source/sukhothai_tambon_web.geojson --out data/`
4. Commit `data/` and push. Pages updates within a minute.

Demo data: `python scripts/compute_indices.py --mock --boundaries data/source/sukhothai_tambon_web.geojson --out data/`

Requires Python 3.10+, pandas, numpy, scipy.

## Publish on GitHub Pages

Settings → Pages → Source: *Deploy from a branch* → `main` / root. Preview locally with `python -m http.server` (opening the file directly won't load `data/`).

## Security

- Never commit keys, tokens or service-account JSON. `.gitignore` excludes common credential files, but check `git status` before each commit.
- If you later automate updates with GitHub Actions + Earth Engine, store the service-account key only in **GitHub Secrets**.

## Data notes and licences

- Boundaries: Royal Thai Survey Department via HDX COD-AB (Thailand admin level 3). Some mirrors list the use limitation as *humanitarian use only*: confirm before publishing in a **public** repository, or use official boundaries from GISTDA / the province.
- DDPM open data is incomplete (FY2564–2565 sparse, FY2558–2561 missing); absence of a record does not mean no drought.
- Index weights and alert thresholds are provisional and must be calibrated against the FY2562–2563 declarations and reviewed with RID.

## External drought indices (intake)

Indices computed elsewhere (e.g. the ARDA / สวก. project) go in `raw/external/<index_id>/` as `data.csv` + `meta.json`.
Spec and template: `intake/INTAKE.md`, `intake/_template/`. Check a delivery with
`python scripts/external_intake.py --check raw/external/<index_id>`. They show on the dashboard and are back-tested
automatically; they only affect alert levels after the team sets `use_in` to `agri` or `water`.
