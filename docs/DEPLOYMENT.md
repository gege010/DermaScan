# Deployment and operations

## Boundary

`docker compose up --build` starts the education-mode baseline, with a loopback-bound web proxy and a private API. Public diagnosis is outside the supported release scope. The application needs no external credentials. Existing `.env` is ignored; old LLM keys are unused.

## Ingress

Use a trusted TLS ingress. Preserve body/time/rate limits and same-origin routing. Restrict API access to the proxy; never forward arbitrary client headers as trusted identity. Default API rate limiting is process-local and conservative (all proxy traffic shares its bucket). Nginx limits by its socket peer; behind another proxy, real-IP trust must be configured only for known ingress addresses. Multi-instance deployments need a shared rate-limit store or upstream enforcement. Limits are abuse controls, not authentication.

All production responses use no-store to avoid caching personal results. Security headers include CSP, no framing, no MIME sniffing, and no referrers. The only inline-style allowance is for React style properties; no inline scripts are permitted. Fonts ship locally. No telemetry by default. Preserve these controls when adding CDN or ingress layers. Resolve maintained base-image tags to tested digests in the release process; the local baseline tracks patched Python, Node and stable nginx images.

## Uploads and compute

The proxy caps requests at 11 MB; the app caps images at 10 MB and 20 MP. Multipart overhead is bounded by the ingress. Direct ASGI exposure without an ingress is unsupported. Application concurrency is bounded; model execution is serialized to protect shared native runtime state. A request timeout cannot kill a hung native inference thread: configure process/container watchdogs and resource limits. Runtime TensorFlow deployments need a dedicated tested ML image and larger memory; the default 768 MB API image is education-only.

Temporary upload spooling uses the container's bounded tmpfs; no persistent disk volume for images. Disable request-body capture, tracing of symptom payloads and proxy body buffering. Do not enable infrastructure request recording on these endpoints. Raw photos, metadata, notes and model scores should not appear in analytics/logs. Encryption at rest, retention and user access design are required before adding server-side health records.

## Health

- `/health`: process alive, model mode and explicit availability; not proof of classifier readiness.
- `/ready`: ready for the supported education mode; `REQUIRE_MODEL=true` makes missing model a 503.
- `/api/v1/model`: research availability and clinical-validation flag.

Monitor latency, status-code counts, resource exhaustion and model availability without retaining personal payloads. Alert on readiness failures and error-rate increases. Review dependencies regularly; Python runtime pins and `web/package-lock.json` make current direct versions reproducible. Lock transitive Python dependencies with platform-specific hashes for a release image, produce an SBOM, and scan built images before promotion.

## Backup and rollback

No server records to back up. Journal is browser-local and users export their own copies; deleting browser data removes persisted notes. Back up trusted model artifacts outside Git with hash/version metadata. Roll back application image and artifact set together. Default-off legacy model can be disabled without losing educational functionality.

## Still required before public medical launch

Clinical validation and clinician content review, threat model and penetration review, jurisdiction-specific medical/privacy assessment, restore/load/soak tests, incident/on-call process, accessibility review with real assistive technology, representative device testing, and monitored infrastructure. Software test success is not clinical approval.
