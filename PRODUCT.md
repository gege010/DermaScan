# DermaScan product and evidence

## Intended experience

The owner's brief: rebuild the existing portfolio into a distinctive web application for understanding skin concerns and supporting care through education. Replace Streamlit, retain the image-analysis/interpretation concept, improve the ML design and commit changes to the existing GitHub repository.

## Capability boundary

The current artifact is a software foundation for skin education and research image analysis, not a validated medical device. Neither old test-set accuracy nor high model scores establish safety for public diagnosis. The interface must never claim to be a doctor, confirm malignancy, rule out disease, prescribe medication, or promise a cure. No synthetic patient records or pretend analysis results.

## Clinical evidence

- Routine education: [AAD skin care basics](https://www.aad.org/public/everyday-care/skin-care-basics) and the individual links in `content/education.json`.
- Serious blistering rash: [NHS Stevens–Johnson syndrome](https://www.nhs.uk/conditions/stevens-johnson-syndrome/).
- Changing lesions: [AAD skin cancer](https://www.aad.org/public/diseases/skin-cancer).
- Preprocessing: [Keras EfficientNet](https://keras.io/api/applications/efficientnet/efficientnet_models/) expects float pixels 0–255 with internal rescaling.
- Leakage risk: [HAM10000 publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/) describes multiple images of the same lesion.

Symptom escalation rules are conservative educational rules, not validated triage. They must take precedence over research image outputs and remain visible immediately in the browser even if the API is unavailable. Unselected flags never mean a person is safe. Content is not yet reviewed by a DermaScan clinician.

## Privacy and retention

No accounts, analytics, remote fonts, external image providers or automatic image retention. The server transiently processes uploads; multipart handling can spool temporarily, so containers use bounded tmpfs. Reverse-proxy request buffering is disabled. No photo persistence or external LLM requests. No request bodies or symptom payloads are logged by the app. Infrastructure operators must preserve that rule.

Journals are session-only by default. Browser persistence is explicit opt-in and is not separately encrypted. This is disclosed before opt-in. Local notes have export, edit and confirmed delete controls. No cloud sync. Internal navigation preserves active drafts; reload loses session data. Future identity, cloud retention and medical-record storage require a separately reviewed design.

## Model operating policy

Default mode is education without a classifier. The old mixed-domain 16-class artifact is opt-in research only. A new compact pipeline trains a linear classifier on frozen ImageNet EfficientNetB0 embeddings, using explicit patient/lesion-disjoint train/validation/test partitions and duplicate-content rejection. It is an experiment in data efficiency, not demonstrated improvement until trained and externally tested. Domain is explicit in the manifest. Unsupported domain yields no classification. No arbitrary confidence threshold is called clinically calibrated.

## Release boundary

Before a public medical launch: expert-reviewed scope and educational content; representative external prospective validation across devices/skin tones; calibrated abstention and OOD evaluation; documented sensitivity/specificity and harms; privacy/security/regulatory assessment; deployment load and recovery testing; incident management. This repository does not claim those gates are complete.
