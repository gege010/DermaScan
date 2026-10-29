---
version: alpha
name: DermaScan
description: A quiet Indonesian skin education studio with an interactive anatomical specimen.
colors:
  primary: "#203b53"
  primary-hover: "#142b3e"
  background: "#fafbfc"
  surface: "#f0f3f5"
  muted: "#526471"
  border: "#dbe2e7"
  blue-wash: "#e7eef3"
  danger: "#9b3636"
  danger-background: "#fcf0ee"
  warning: "#825b26"
typography:
  display:
    fontFamily: "Manrope Variable, Segoe UI, sans-serif"
  body:
    fontFamily: "DM Sans Variable, Segoe UI, sans-serif"
rounded:
  DEFAULT: "12px"
  panel: "22px"
  control: "7px"
spacing:
  section-gap: "100px"
  page-max: "1248px"
components:
  button:
    rounded: "7px"
  panel:
    rounded: "12px"
---

# DermaScan design system

## Overview

An Indonesian skin education studio for adults who want to understand a concern before choosing a next step. Hybrid register: the home page introduces a point of view, while analysis, knowledge and journal pages prioritize comprehension and honest feedback.

The signature is an exploded, interactive skin specimen. Its materials borrow from anatomical teaching models, with ink-blue annotation and warm tissue layers. All diagrams are explicitly illustrative. Avoid hospital dashboards, beauty-product commerce, invented clinical trust badges, gradient hero blobs, oversized pills, and medical outcome promises.

References supplied by the user: impeccable.style (restraint and critique), tasteskill.dev (intentional typography), emilkowal.ski/skill and ui-skills.com/skills/emilkowalski/apple-design (immediate feedback, restrained motion, reduced motion), uupm.cc (state coverage). These are inspiration, not imported code or business policy.

## Colors

Runtime owner: `web/src/styles.css :root`. The values above document the accepted palette; runtime CSS is the canonical source. Map primary to `--ink`, primary-hover to `--ink-hover`, background to `--paper`, surface to `--surface`, muted to `--muted`, border to `--line`, blue-wash to `--blue-wash`, danger to `--danger`, danger-background to `--danger-bg`, warning to `--warning`. Shared primitives consume these variables; pages do not invent new brand palettes. Anatomical and article illustration colors are intentionally local illustrative materials, not semantic state colors.

## Typography

Self-hosted Manrope Variable for headings, DM Sans Variable for body and utility. CSS `--font-display` and `--font-body` are the only family owners. Heading scale: home 46–80px, page 41–55px, section 27–39px; body 12–16px, captions 9–11px. Tight display tracking is paired with generous reading line-height. Indonesian is the only supported locale; do not introduce untranslated product controls.

## Layout

1248px maximum canvas with 56px desktop gutters, 22px phone gutters. Natural document scrolling, sticky 88px desktop / 68px phone header. Home is an asymmetrical two-column composition collapsing to one column; app forms use a main column and quiet help rail. Reading pages cap line length. Knowledge uses six bounded articles and URL-restorable local filters; journal has a 100-entry cap. No unbounded or viewport-locked tables.

## Elevation & Depth

Content is flat. Borders organize tasks. Elevation belongs to illustration materials and the app-owned confirmation dialog. No card hover lift. Blur is limited to header and modal backdrop.

## Shapes

7px controls, 12px working surfaces, 22px dialog/results. One 72px cut at the home specimen's bottom-right is a deliberate specimen-board signature. Do not spread this cut to unrelated screens.

## Components

`web/src/components/ui.tsx` owns Brand, ArrowLink, SectionHead, Notice, Dialog and Art. `SkinStudy.tsx` owns the interactive specimen. CSS owns shared button/choice/input recipes, including hover, focus, active, disabled and busy states. Selected choices use both native input state and visible borders. Focus remains visible; `main:focus` alone suppresses the programmatic route-focus outline.

Motion: 150–200ms color/press feedback, 300–650ms specimen changes; no fake analysis timers. Reduced motion removes transforms and uses a static labeled busy state. Scrollbars inherit global thumb/track/hover/active tokens, including forced-colors fallback.

## Do's and Don'ts

- Explain what a person can do next. Never present a research softmax score as a diagnosis probability.
- Keep unsupported and unavailable outcomes as first-class results.
- Never add decorative dashboards, fabricated testimonials, clinician approval, health scores or streak pressure.
- Keep source links and privacy descriptions close to the action.
- Keep every article, navigation element and button functional.
