# DermaScan UX contract

Business authority: `PRODUCT.md`; API behavior: `deployment/api/main.py`; visual owner: `DESIGN.md` and `web/src/styles.css`.

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Form | native fieldset/label + shared CSS choice and input recipes | PRODUCT.md, API enums | analysis, journal | browser tests, API tests |
| Scrollbar | global styles.css baseline | DESIGN.md | geometry only | static audit + browser |
| Toast | inline Notice / role=status regions | PRODUCT.md | info, warning, danger, saved | browser tests |
| CRUD | Journal + SessionProvider + storage.ts | PRODUCT.md privacy | session or opt-in browser | create/edit/delete/export browser tests |
| Dialog | shared ui.tsx Dialog using HTML dialog | PRODUCT.md privacy | delete note, disable persistence | focus, Escape and restore browser checks |
| Upload | Scan + imaging.py | API image contract | clinical, dermoscopy | validation, abort, preview, API tests |
| Search | Learn URL parameters | bounded knowledge catalog | local instant search | clear, filter, no-results browser tests |

Select/Listbox, Date and Table Selection are not applicable: choices use small native radio groups, timestamps are generated, and no tables are selectable.

## Routes and state

- `/`: subject-led introduction, real layer selection, links to the three tasks.
- `/analisis`: one photo, optional symptoms, explicit consent, contextual assessment. File/draft/result live only in SessionProvider and survive internal navigation. File URLs are revoked. No sample diagnosis is generated when a model is missing.
- `/panduan`: six source-linked guides, local filter/search in `q` and `kategori` URL parameters; immediate clear restores focus. No remote search race.
- `/panduan/:id`: readable guide; unknown id has a working return link.
- `/jurnal`: session notes, explicit browser-storage opt-in, edit, confirmed delete, text export. Maximum 100 notes / 2000 characters each. Persistence errors preserve existing notes and draft. No false success on storage failure.
- `/pendekatan`: capability, model, sources, privacy, emergency boundaries.
- unknown route: app-owned missing page.

## Flow ledger

| Action | Outcome | Failure/recovery | Source |
|---|---|---|---|
| Choose photo | local preview, no upload yet | retain previous valid photo; inline error | PRODUCT.md privacy, imaging.py |
| Analyze | one request, stable busy button, focus result | abort/timeout/offline preserve photo and context; retry explicit | API contract |
| Serious symptoms | immediate urgent direction, photo not required | visible before API request; no routine care on urgent result | PRODUCT.md clinical evidence |
| Edit input | invalidate earlier result | cannot display stale analysis as current | API context contract |
| Add/edit journal | update owning list in place; status text | preserve draft on storage failure | PRODUCT.md privacy |
| Delete journal | app-owned confirmation, explicit object/consequence | cancel/Escape leaves note; focus restored | PRODUCT.md privacy |
| Disable storage | confirmation deletes browser copy; in-memory notes remain | error preserves enabled state | PRODUCT.md privacy |

No native alert/confirm/prompt. Dialog is app-owned HTML `<dialog>` with title, native focus containment/inert background/Escape and focus restoration; cancel is initial focus. Navigation uses native links, actions use buttons. All product forms use noValidate and associated text errors. No hidden scrollbars, unactionable links, fake percent progress or forced health streaks. Language id-ID; timestamps use the user's browser time zone and native Indonesian formatting. Single light theme; reduced-motion and high-contrast fallbacks.
