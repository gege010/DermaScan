# Verification record — 29 October 2026

## Executed successfully

| Check | Result |
|---|---|
| `python -m pytest -q` | 26 passed: API, images, model adapter and training manifest |
| `npm --prefix web test` | 30 passed across desktop Chromium and phone Chromium |
| axe checks | No reported WCAG A/AA violations on home, analysis, library, journal, approach, and the unsupported-result workflow |
| Web `build`, `typecheck`, `format:check` | Passed |
| Ruff lint and format checks on new Python modules/tests | Passed |
| `pip check` | Passed |
| `pip-audit -r requirements.txt --progress-spinner off` | No known vulnerabilities reported after runtime upgrades |
| npm dependency audit during installation | No vulnerabilities reported |
| Premium UI strict static audit | No findings |
| Official DESIGN.md lint | Zero errors; nonblocking orphan-token warnings because runtime token mapping is documented in prose |
| `docker compose config --quiet` | Passed |
| `git diff --check` | Passed |
| `python -m scripts.smoke_legacy_model` in isolated ML runtime | Real local HDF5 artifact loaded via TensorFlow 2.20 / tf-keras 2.20.1; input 300×300×3, 16 finite output scores, correct 0–255 pixel contract |

Browser tests cover layer selection, responsive navigation and overflow, source-linked reading, local search/filter/no-results recovery, explicit consent, symptom-first direction, failed requests and retry, upload preview and busy state, unsupported results, journal handoff, create/edit/export/delete, opt-in persistence, keyboard focus and reduced motion. The successful-result browser test uses a fixture captured from the real API with a synthetic image; it is a UI contract test, not a diagnostic benchmark.

Manual review in the in-app browser covered the home composition, analysis form, narrow 390px layout and a real API-backed direct-evaluation result without a photo. Local desktop/phone screenshots are generated in ignored `artifacts/` by the browser suite. Tests do not use real patient photos.

## Verification limits

- Docker Desktop's start command was attempted, but its Linux engine pipe remained unavailable. Container image build, nginx execution, container health and Compose startup are **not runtime-verified** on this machine. The Compose file itself validates.
- The compact classifier pipeline has not been trained or evaluated on clinical data in this revision. No new accuracy figure or improvement claim is made.
- Unit tests use a fake model to verify modality routing and malformed-output handling. These are not ML accuracy tests.
- Historical model metrics were inspected, not reproduced. True clinical validation, calibration, OOD rejection, subgroup evaluation and clinician content review remain required.
- The real-artifact smoke check uses only a synthetic checker pattern. It establishes serialization/tensor compatibility, not classification quality, and does not activate the model in the default education deployment.
- Automated axe success is not a complete assistive-technology audit. Physical phone devices and screen readers have not been tested.
- One nonblocking upstream test-client deprecation warning remains (`httpx` compatibility); API tests pass.

GitHub CI is configured to repeat API, formatting, web build, browser, accessibility and runtime dependency checks on push. Local results above do not assert a remote CI result before that run finishes.
