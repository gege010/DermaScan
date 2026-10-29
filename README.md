# DermaScan

A web platform for understanding skin concerns through contextual image assessment, source-linked education, and a private skin journal.

**Status:** working software foundation for education and research. It is **not a clinically validated diagnostic or treatment service**. No reported accuracy is presented as a guarantee for real-world photos.

## What changed in v3

- React + TypeScript web interface replaces Streamlit: interactive anatomy, responsive analysis workflow, searchable education library, journal, and capability/privacy page.
- FastAPI works without TensorFlow or external API keys. No image is sent to Groq, Tavily, or another external AI provider.
- Serious symptom guidance takes precedence over image inference. Missing models, poor images, unsupported domains, and errors produce explicit outcomes rather than fabricated diagnoses.
- Canonical RGB preprocessing, decoded-image validation, EXIF orientation, file/pixel limits, bounded inference concurrency, readiness, rate limits, and hardened same-origin proxy.
- Session-only journal by default; optional browser persistence, edit, export, and confirmed deletion. No medical data is stored in Git or a server database.
- Compact research pipeline: frozen EfficientNetB0 embeddings + linear classification, explicit group-disjoint splits and duplicate-content checks.

## Run locally

Python 3.11 and Node.js 22.12+ (or Node 24 LTS).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m uvicorn deployment.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In a second terminal:

```powershell
cd web
npm ci
npm run dev
```

Open [the web app](http://127.0.0.1:5173). Vite proxies `/api` to the local API. No `.env` is required in education mode. Optional settings are documented in `.env.example`. Existing Groq/Tavily keys are not used by v3. Self-hosted fonts; no CDN dependency at runtime.

On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python`.

## Containers

```sh
docker compose up --build
```

Open [DermaScan](http://localhost:8080). Only the web proxy is exposed, bound to loopback. The API remains on the private Compose network. Both containers use non-root users, read-only filesystems, bounded temporary filesystems and health checks. Uploads are transient, including framework multipart buffers. The default image intentionally excludes TensorFlow; do not turn model flags on without installing a compatible ML runtime.

Before Internet deployment, terminate TLS at a trusted ingress, preserve request size/rate/time limits and security headers, configure resource limits and monitoring, and follow [deployment notes](docs/DEPLOYMENT.md). Do not expose the API port directly. The Compose configuration is a local deployment baseline, not a claim that public medical-launch gates have passed.

## Optional image research

### Existing 16-class model

The old HDF5 artifact named `skin_model_best.keras` is retained locally and ignored by Git. Enable only in a dedicated local research environment:

```powershell
.venv/Scripts/python -m pip install -r requirements-ml.txt
.venv/Scripts/python -m pip install tf-keras==2.20.1
$env:ENABLE_LEGACY_MODEL = 'true'
.venv/Scripts/python -m uvicorn deployment.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Keep the original ordered `class_names.json` beside the model. Keras 2 HDF5 loading may require `tf-keras`; the loader falls back explicitly. Startup failure is reported as model-unavailable mode. This model mixes dermoscopic and clinical images and has known split, class balance and domain limitations. Raw scores are available only as **research scores**, not diagnosis probabilities. Domain mismatch is not repaired by renormalizing scores. There is no validated skin/non-skin detector, OOD guarantee, or clinical confidence threshold.

### Less training data: frozen embeddings

Prepare a curated CSV with **real** patient/lesion group identifiers and fixed partitions:

```csv
image_path,label,group_id,split
photos/a.jpg,acne,patient-a,train
photos/b.jpg,acne,patient-b,val
photos/c.jpg,acne,patient-c,test
```

Every partition must contain the same two or more classes. Paths are relative to the CSV. Never invent per-image patient identifiers to make checks pass. Complete manifests must be deduplicated; perceptual duplicates and source authenticity still require review.

```sh
python -m scripts.train_compact --csv data/curated.csv --domain clinical --max-per-class 50 --output models/compact
```

ImageNet backbone weights may download during training. `--max-per-class` caps only training data; validation/test sets stay untouched. Validation selects regularization; the held-out test is evaluated once. No clinical improvement is claimed until measured on representative external data. The output includes backbone, classifier, manifest and evaluation. Point `MODEL_DIR` at the operator-controlled output directory to load it. Joblib files are trusted code artifacts; never accept them from end users.

Original notebooks, `src/train.py` and `scripts/run_domain_eval.py` are retained as **historical experiments**, not the v3 deployment pipeline. Their old image-level split does not establish lesion-disjoint performance. See [model card](docs/MODEL_CARD.md).

For an optional real-artifact compatibility check, run `python -m scripts.smoke_legacy_model` in the ML environment. It uses only a synthetic checker pattern to verify loading, shape, pixel range and finite model outputs; it does not evaluate medical accuracy.

## Verify

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m pip check
cd web
npm run format:check
npm run build
npx playwright install chromium
npm test
```

Browser tests cover desktop and phone layouts, axe accessibility checks, layer interaction, search/filter recovery, consent, emergency guidance, analysis failure, journal create/edit/delete/export, opt-in persistence, focus restoration, and reduced motion. API tests use synthetic images and cover upload validation, missing model, unsafe-label escalation, invalid inputs, consent, rate limits, and group leakage. They do not establish clinical accuracy.

CI runs API and web checks. [Verification notes](docs/VERIFICATION.md) distinguish executed checks from remaining release gates.

## Structure

```text
web/                   React UI, shared primitives, browser tests
content/education.json Source-linked education shared by API and UI
src/dermascan/         Image contract, assessment policy, model adapter
deployment/api/       FastAPI application
deployment/web/       Hardened same-origin reverse proxy
scripts/train_compact.py  Group-aware, small-data research training
tests/                 API/image/manifest tests
docs/                  Model card, deployment and verification
```

Product and design decisions: [PRODUCT.md](PRODUCT.md), [DESIGN.md](DESIGN.md), [UX-CONTRACT.md](UX-CONTRACT.md).

## Clinical and privacy boundaries

For adult education. No diagnosis, medication prescription, severity scoring, or cure claim. Severe symptoms require direct medical attention; a negative or unavailable image result must never delay care. Educational summaries have source links but are not yet reviewed by a DermaScan clinician. No cloud journal storage, account system, or monitoring by doctors. Browser notes are not separately encrypted and can be read by other users of the same browser.

See [AAD skin care basics](https://www.aad.org/public/everyday-care/skin-care-basics), [NHS serious skin reactions](https://www.nhs.uk/conditions/stevens-johnson-syndrome/), and [the project model card](docs/MODEL_CARD.md).
