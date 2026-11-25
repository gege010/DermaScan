# DermaScan

[![Verify DermaScan](https://github.com/gege010/DermaScan/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gege010/DermaScan/actions/workflows/ci.yml)

DermaScan helps Indonesian-speaking adults understand and track skin concerns. Users can describe symptoms, add an optional photo, read dermatology guides with links to their sources, and keep a journal of changes to discuss at a medical appointment.

![DermaScan home page with a button to start an assessment and an interactive atlas of the skin layers](docs/images/beranda.png)

## Features and privacy

- **Symptom assessment.** The app prioritizes guidance for serious symptoms and explains when it cannot assess a photo. Education mode runs without a model, TensorFlow, or API keys.
- **Skin atlas and guides.** Explore three layers of the skin, search articles by topic, filter by category, and follow links to the original sources.
- **Personal journal.** Create, edit, export, and delete entries. Entries last for the current session by default. If you enable local storage, anyone using the same browser profile can read them.
- **Photo handling.** The API processes uploaded photos temporarily. It does not archive them or send them to external AI providers.

DermaScan is intended for education and research. The optional model has not been validated for clinical diagnosis. For details on its training data, evaluation, and limitations, see the [model card](docs/MODEL_CARD.md).

## Quick start with Docker

Install Git and Docker with the Compose plugin, then run the following commands.

```sh
git clone https://github.com/gege010/DermaScan.git
cd DermaScan
docker compose up --build --wait
```

Open the [local app](http://127.0.0.1:8080). The default installation runs in education mode without additional configuration. If assessments fail to load, check the [API readiness endpoint](http://127.0.0.1:8080/ready) and run `docker compose logs api web`. To stop the app, run `docker compose down`.

## Local development

Use Python 3.11 and Node.js 22.12+ or 24 LTS. In the first terminal, create a Python environment from the repository root.

```sh
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash. Then install the dependencies and start the API.

```sh
python -m pip install -r requirements-dev.txt
python -m uvicorn deployment.api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In a second terminal, start the React app from the repository root.

```sh
cd web
npm ci
npm run dev
```

Open the [development app](http://127.0.0.1:5173). Vite forwards API requests to port 8000. See the [example environment file](.env.example) for optional settings.

### Testing

With the Python environment active, run the backend tests from the repository root.

```sh
python -m pytest -q
```

Run the frontend checks from the `web/` directory.

```sh
npm run format:check
npm run build
npx playwright install chromium
npm test
```

Playwright starts its own test server on port 5180. Set `DERMASCAN_TEST_PORT` to a free port if needed. For test coverage and verification limits, see the [verification notes](docs/VERIFICATION.md).

## Documentation

| Topic | Resources |
| --- | --- |
| Deployment and reverse proxy | [Deployment guide](docs/DEPLOYMENT.md) |
| Model evaluation and experiments | [Model card](docs/MODEL_CARD.md) · [Training script](scripts/train_compact.py) |
| Product goals and user flows | [Product specification](PRODUCT.md) · [UX contract](UX-CONTRACT.md) |
| Interface design | [Design system](DESIGN.md) |

## Support and contributions

DermaScan is maintained by [@gege010](https://github.com/gege010). To report a problem, include the steps to reproduce it, what you expected to happen, and your browser or runtime version. Use synthetic data in examples and screenshots to protect personal health information. [Open a GitHub issue](https://github.com/gege010/DermaScan/issues).
