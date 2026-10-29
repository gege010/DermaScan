# DermaScan model card

## Intended use

Adult skin education, documentation and controlled ML research. Not diagnosis, screening clearance, prescribing, or emergency triage. No current artifact has prospective clinical validation. The API always reports `clinical_validation: false`.

## Existing artifact

EfficientNetB3, 300×300 RGB input, 16 ordered labels. Locally available artifact begins with an HDF5 signature despite its `.keras` suffix. Original saved metadata reports accuracy 0.8562 and macro F1 0.6114; these are historical image-split research results, not independently reproduced in this revision. Melanoma recall is 0.3393 and combination-skin F1 is 0.0 in saved metrics. Never use aggregate accuracy as a product quality claim.

The notebook calls the additional source SD101, while the old README called it SD-198. Provenance is unresolved; it is not silently relabeled in v3. The preprocessing notebook discarded HAM10000 `lesion_id`; image-random splits may leak related lesions. Type-of-skin and disease labels are mixed in a mutually exclusive classifier even though concerns can coexist. Dermoscopy and consumer photo distributions differ. Very strong source-specific scores are not evidence of cross-device or cross-population generalization.

## Compact research alternative

Frozen ImageNet EfficientNetB0 global-average-pooled embeddings, standardized features and class-balanced logistic regression. Regularization selected on validation macro F1; test split untouched until final reporting. `--max-per-class` allows training-sample experiments. Train/validation/test membership is explicit in the input manifest; checks reject cross-split patient/lesion groups, duplicate file bytes, missing fields and missing classes. Human review must still establish true patient grouping, licensing, labels, near-duplicates and source authenticity.

Artifacts include schema version, domain, label order, input dimensions, preprocessing, source manifest hash and evaluation. Operator must pin/trust artifact provenance. Do not load uploaded model/joblib files. A stable group split is necessary but not sufficient to prove generalization. ImageNet pretraining may be unsuitable for the selected conditions; compare it against a documented baseline before promotion.

## Inference and abstention

- Default model mode is unavailable/education; no synthetic replacement results.
- One EXIF-aware RGB float32 0–255 preprocessing path for API, compact training and CLI.
- File signature, pixel/file limits and basic exposure/detail checks. These do not prove skin presence, anatomical suitability or diagnostic quality.
- Explicit image modality; compact domain mismatch yields no score. Legacy top-label modality conflict also yields no score, without rescaling incompatible scores into a false confidence.
- Serious symptom selections bypass image inference. Research labels `mel`, `bcc`, `akiec`, `SJS-TEN` request direct evaluation rather than a normal urgency fallback.
- All model distributions remain uncalibrated experimental scores. There is no validated OOD detector, probability-of-disease estimate, or safe threshold. No automatic prescriptions or LLM diagnosis text.

## Promotion gates

Independent clinical scope/content review; representative external and prospective evaluation; lesion/patient-disjoint verified provenance; per-class sensitivity/specificity with uncertainty; calibration and abstention/coverage trade-offs; non-skin and unseen disease rejection; device, age and skin-tone subgroup analysis; operational monitoring and rollback; applicable privacy/security/regulatory review. None is claimed complete merely because software tests pass.
