import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Camera,
  Check,
  CheckCircle2,
  FileImage,
  Info,
  LoaderCircle,
  LockKeyhole,
  NotebookPen,
  RotateCcw,
  ShieldCheck,
  Upload,
  X,
} from 'lucide-react';
import { Notice } from '../components/ui';
import { useSession } from '../session';
import type { Assessment } from '../types';

const common = [
  { id: 'itch', label: 'Gatal' },
  { id: 'pain', label: 'Nyeri' },
  { id: 'dryness', label: 'Kering / mengelupas' },
  { id: 'redness', label: 'Kemerahan' },
  { id: 'breakouts', label: 'Jerawat / benjolan' },
];
const important = [
  { id: 'changing_mole', label: 'Tahi lalat berubah atau luka tidak sembuh' },
  { id: 'breathing', label: 'Sulit bernapas atau wajah membengkak' },
  { id: 'blisters_fever', label: 'Kulit melepuh disertai demam atau luka di mulut / mata' },
];
const areas = [
  { id: 'face', label: 'Wajah' },
  { id: 'body', label: 'Tubuh' },
  { id: 'hands', label: 'Tangan / kaki' },
  { id: 'scalp', label: 'Kulit kepala' },
  { id: 'other', label: 'Lainnya' },
];
const durations = [
  { id: 'days', label: 'Beberapa hari' },
  { id: 'weeks', label: 'Beberapa minggu' },
  { id: 'months', label: 'Lebih dari sebulan' },
  { id: 'unknown', label: 'Belum yakin' },
];
const labels: Record<string, string> = {
  urgent: 'Cari pertolongan segera',
  review: 'Perlu pemeriksaan langsung',
  retake: 'Foto perlu diulang',
  unsupported: 'Belum dapat disimpulkan',
  research: 'Hasil eksperimen',
};

export default function Scan() {
  const { draft, setDraft, result, setResult, setJournalDraft } = useSession();
  const [preview, setPreview] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [fileError, setFileError] = useState('');
  const [consentError, setConsentError] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const consentRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const controller = useRef<AbortController | null>(null);
  const navigate = useNavigate();
  const urgent = draft.symptoms.some((s) => ['breathing', 'blisters_fever'].includes(s));
  const review = draft.symptoms.includes('changing_mole');
  useEffect(() => {
    if (!draft.file) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(draft.file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [draft.file]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (result) {
      resultRef.current?.focus({ preventScroll: true });
      resultRef.current?.scrollIntoView({ block: 'start', behavior: 'instant' });
    }
  }, [result]);
  function change(patch: Partial<typeof draft>) {
    setDraft((d) => ({ ...d, ...patch }));
    setResult(null);
    setError('');
  }
  function selectFile(file?: File) {
    if (!file) return;
    if (
      !['image/jpeg', 'image/png', 'image/webp'].includes(file.type) ||
      file.size > 10 * 1024 * 1024 ||
      !file.size
    ) {
      setFileError('Kamu bisa mengunggah JPG, PNG, atau WebP hingga 10 MB. Pilih foto lain.');
      return;
    }
    setFileError('');
    change({ file });
  }
  function toggle(id: string) {
    change({
      symptoms: draft.symptoms.includes(id)
        ? draft.symptoms.filter((s) => s !== id)
        : [...draft.symptoms, id],
    });
  }
  async function analyze(e: React.SubmitEvent) {
    e.preventDefault();
    if (busy) return;
    if (!draft.file && !urgent && !review) {
      setFileError('Pilih satu foto kulit untuk melanjutkan.');
      input.current?.focus();
      return;
    }
    if (!draft.consent) {
      setConsentError(true);
      consentRef.current?.focus();
      return;
    }
    setError('');
    setResult(null);
    setBusy(true);
    const abort = new AbortController();
    controller.current = abort;
    const timeout = window.setTimeout(() => abort.abort('timeout'), 45000);
    const body = new FormData();
    body.set('consent', 'true');
    body.set('image_kind', draft.imageKind);
    body.set('symptoms', JSON.stringify(draft.symptoms));
    body.set('duration', draft.duration);
    body.set('area', draft.area);
    if (draft.file) body.set('file', draft.file);
    try {
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        body,
        signal: abort.signal,
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Data belum lengkap. Periksa pilihanmu dan coba lagi.',
        );
      if (!abort.signal.aborted) setResult(data as Assessment);
    } catch (cause) {
      if (abort.signal.aborted && abort.signal.reason !== 'timeout') return;
      setError(
        abort.signal.aborted
          ? 'Analisis membutuhkan waktu terlalu lama. Coba lagi; foto dan pilihanmu masih tersedia.'
          : cause instanceof TypeError
            ? 'Layanan belum terhubung. Periksa koneksi dan coba lagi. Foto dan pilihanmu tetap tersedia.'
            : cause instanceof Error
              ? cause.message
              : 'Analisis belum berhasil. Coba lagi.',
      );
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  }
  function journal() {
    if (!result) return;
    setJournalDraft(
      `Keluhan: ${result.context.symptoms.join(', ') || 'Belum dicatat'}. ${result.assessment.title}.`,
    );
    navigate('/jurnal');
  }

  return (
    <div className="container page scan-page">
      <div className="page-intro">
        <p className="eyebrow">KENALI KULIT</p>
        <h1>Mulai dari ceritamu.</h1>
        <p>Foto memberi gambaran. Apa yang kamu rasakan memberi konteks.</p>
      </div>
      <div className="scan-layout">
        <form noValidate onSubmit={analyze} className="scan-form" aria-busy={busy}>
          <fieldset disabled={busy} className="bare-fieldset">
            <section className="form-section">
              <div className="form-section-heading">
                <span className="step-number">01</span>
                <div>
                  <h2>Satu foto, lebih dekat.</h2>
                  <p>Ambil bagian kulit yang ingin kamu pahami.</p>
                </div>
              </div>
              <div
                className={`upload-zone ${preview ? 'has-preview' : ''} ${dragging ? 'dragging' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!busy) setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  if (!busy) selectFile(e.dataTransfer.files[0]);
                }}
              >
                {preview ? (
                  <>
                    <img src={preview} alt="Pratinjau foto kulit yang dipilih" />
                    <div className="preview-controls">
                      <span>
                        <FileImage size={15} />
                        {draft.file?.name}
                      </span>
                      <button
                        type="button"
                        className="icon-button"
                        onClick={() => {
                          change({ file: null });
                          if (input.current) input.current.value = '';
                        }}
                        aria-label="Hapus foto"
                      >
                        <X size={18} />
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <span className="upload-symbol">
                      <ScanFrame />
                    </span>
                    <h3>Ruang untuk foto kulitmu</h3>
                    <p>Letakkan foto di sini, atau pilih dari perangkat.</p>
                  </>
                )}
                <label
                  className={`button ${preview ? 'upload-replace' : 'secondary'}`}
                  htmlFor="skin-photo"
                >
                  <Upload size={16} />
                  {preview ? 'Ganti foto' : 'Pilih foto'}
                </label>
                <input
                  ref={input}
                  id="skin-photo"
                  className="visually-hidden file-input"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  aria-invalid={!!fileError}
                  aria-describedby="photo-help photo-error"
                  onChange={(e) => selectFile(e.target.files?.[0])}
                />
                {!preview && (
                  <span className="upload-format">JPG, PNG, WebP · Maks. 10 MB · Maks. 20 MP</span>
                )}
              </div>
              <p id="photo-error" className="field-error" role={fileError ? 'alert' : undefined}>
                {fileError}
              </p>
              <p id="photo-help" className="field-help">
                <Camera size={15} />
                Cahaya alami, tanpa filter, fokus pada area kulit. Hindari identitas pribadi di
                foto.
              </p>
              <fieldset className="choice-fieldset">
                <legend>Jenis foto</legend>
                <div className="radio-options">
                  {[
                    { id: 'clinical', label: 'Foto kamera biasa' },
                    { id: 'dermoscopy', label: 'Foto dermoskopi' },
                  ].map((v) => (
                    <label
                      className={`choice ${draft.imageKind === v.id ? 'selected' : ''}`}
                      key={v.id}
                    >
                      <input
                        type="radio"
                        name="kind"
                        value={v.id}
                        checked={draft.imageKind === v.id}
                        onChange={() => change({ imageKind: v.id })}
                      />
                      {v.label}
                    </label>
                  ))}
                </div>
                <p className="field-help">
                  Dermoskopi diambil dengan alat pembesar khusus; bukan mode zoom kamera.
                </p>
              </fieldset>
            </section>
            <section className="form-section">
              <div className="form-section-heading">
                <span className="step-number">02</span>
                <div>
                  <h2>Apa yang kamu rasakan?</h2>
                  <p>Pilih yang sesuai. Tidak apa-apa jika belum yakin.</p>
                </div>
              </div>
              <fieldset className="choice-fieldset">
                <legend>Area kulit</legend>
                <div className="radio-options">
                  {areas.map((v) => (
                    <label key={v.id} className={`choice ${draft.area === v.id ? 'selected' : ''}`}>
                      <input
                        type="radio"
                        name="area"
                        checked={draft.area === v.id}
                        onChange={() => change({ area: v.id })}
                      />
                      {v.label}
                    </label>
                  ))}
                </div>
              </fieldset>
              <fieldset className="choice-fieldset">
                <legend>
                  Keluhan yang muncul <span className="optional">· boleh lebih dari satu</span>
                </legend>
                <div className="radio-options">
                  {common.map((v) => (
                    <label
                      key={v.id}
                      className={`choice ${draft.symptoms.includes(v.id) ? 'selected' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={draft.symptoms.includes(v.id)}
                        onChange={() => toggle(v.id)}
                      />
                      {draft.symptoms.includes(v.id) && <Check size={13} />} {v.label}
                    </label>
                  ))}
                </div>
              </fieldset>
              <fieldset className="choice-fieldset">
                <legend>Sudah berapa lama?</legend>
                <div className="radio-options">
                  {durations.map((v) => (
                    <label
                      key={v.id}
                      className={`choice ${draft.duration === v.id ? 'selected' : ''}`}
                    >
                      <input
                        type="radio"
                        name="duration"
                        checked={draft.duration === v.id}
                        onChange={() => change({ duration: v.id })}
                      />
                      {v.label}
                    </label>
                  ))}
                </div>
              </fieldset>
              <fieldset className="important-symptoms">
                <legend>Apakah ada tanda berikut?</legend>
                {important.map((v) => (
                  <label key={v.id} className="check-row">
                    <input
                      type="checkbox"
                      checked={draft.symptoms.includes(v.id)}
                      onChange={() => toggle(v.id)}
                    />
                    <span>{v.label}</span>
                  </label>
                ))}
                <p>Gejala ini lebih penting daripada hasil foto.</p>
              </fieldset>
              {urgent && (
                <Notice tone="danger">
                  <strong>Jangan menunggu analisis.</strong>
                  <p>
                    Cari pertolongan di IGD terdekat sekarang. Minta bantuan orang terdekat. Foto
                    tidak diperlukan untuk menindaklanjuti gejala ini.
                  </p>
                </Notice>
              )}
              {review && !urgent && (
                <Notice tone="warning">
                  Atur pemeriksaan langsung dengan dokter kulit. Hasil foto tidak bisa memastikan
                  penyebab perubahan ini.
                </Notice>
              )}
            </section>
            <section className="form-section consent-section">
              <div className="form-section-heading">
                <span className="step-number">03</span>
                <div>
                  <h2>Lanjutkan dengan sadar.</h2>
                  <p>Pahami bagaimana fotomu digunakan.</p>
                </div>
              </div>
              <label className="check-row consent">
                <input
                  ref={consentRef}
                  type="checkbox"
                  checked={draft.consent}
                  aria-invalid={consentError}
                  aria-describedby="consent-error"
                  onChange={(e) => {
                    change({ consent: e.target.checked });
                    setConsentError(false);
                  }}
                />
                <span>
                  Saya setuju foto diproses untuk analisis ini. Saya memahami bahwa hasilnya
                  bersifat edukatif, belum tervalidasi klinis, dan bukan diagnosis dokter.{' '}
                  <Link to="/pendekatan">Baca kemampuan & privasi.</Link>
                </span>
              </label>
              <p id="consent-error" className="field-error">
                {consentError ? 'Centang persetujuan untuk melanjutkan.' : ''}
              </p>
              <button className="button primary analyze-button" type="submit" disabled={busy}>
                {busy ? (
                  <>
                    <LoaderCircle size={18} className="spinner" />
                    Memproses analisis
                  </>
                ) : (
                  <>
                    {urgent || review ? 'Lihat langkah berikutnya' : 'Pahami kulitku'}
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </section>
          </fieldset>
          {busy && (
            <div className="busy-state" role="status">
              <p>Memeriksa foto dan menyiapkan panduan…</p>
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  controller.current?.abort();
                  setBusy(false);
                }}
              >
                Batalkan analisis
              </button>
            </div>
          )}
          {error && (
            <div className="inline-error" role="alert">
              <Info size={20} />
              <p>{error}</p>
              <button type="submit" className="text-button">
                Coba lagi
              </button>
            </div>
          )}
        </form>
        <aside className="scan-aside">
          <div className="aside-panel">
            <span className="eyebrow">SEBELUM MEMULAI</span>
            <h3>
              Hasil yang jujur
              <br />
              lebih berarti.
            </h3>
            <p>
              DermaScan membantumu menyiapkan langkah berikutnya, bukan memastikan penyakit lewat
              satu foto.
            </p>
            <ul className="trust-list">
              <li>
                <ShieldCheck size={19} />
                <span>Tanpa diagnosis yang dipaksakan</span>
              </li>
              <li>
                <LockKeyhole size={19} />
                <span>Foto tidak disimpan oleh aplikasi</span>
              </li>
              <li>
                <CheckCircle2 size={19} />
                <span>Penjelasan dengan sumber terbuka</span>
              </li>
            </ul>
            <Link className="arrow-link" to="/pendekatan">
              Kenali batas analisis
              <ArrowUpRight size={16} />
            </Link>
          </div>
          <div className="photo-tips">
            <p className="eyebrow">FOTO YANG MEMBANTU</p>
            <div className="photo-tip-frame">
              <div className="tip-skin" />
              <span className="corner tl" />
              <span className="corner tr" />
              <span className="corner bl" />
              <span className="corner br" />
            </div>
            <p>
              Area terlihat jelas.
              <br />
              Jarak cukup dekat, cahaya merata.
            </p>
            <span>Ilustrasi, bukan contoh diagnosis.</span>
          </div>
        </aside>
      </div>
      {result && (
        <section
          ref={resultRef}
          tabIndex={-1}
          className="analysis-result"
          aria-label="Hasil analisis"
        >
          <div className="result-topline">
            <p className="eyebrow">LANGKAH BERIKUTNYA</p>
            <span className={`status-badge ${result.assessment.status}`}>
              {labels[result.assessment.status]}
            </span>
          </div>
          <div className="result-summary">
            <h2>{result.assessment.title}</h2>
            <p>{result.assessment.summary}</p>
          </div>
          <div className="result-grid">
            <div>
              <h3>
                {result.assessment.status === 'retake'
                  ? 'Perbaiki fotomu'
                  : 'Yang bisa kamu lakukan'}
              </h3>
              <ol className="next-steps">
                {result.assessment.next_steps.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ol>
              {result.education.map((item) => (
                <div className="education-block" key={item.title}>
                  <h4>{item.title}</h4>
                  <p>{item.body}</p>
                </div>
              ))}
            </div>
            <div className="result-evidence">
              <h3>Di balik hasil ini</h3>
              <dl>
                <div>
                  <dt>Keluhan</dt>
                  <dd>{result.context.symptoms.join(', ') || 'Tidak ada yang dipilih'}</dd>
                </div>
                <div>
                  <dt>Pemeriksaan foto</dt>
                  <dd>
                    {result.quality
                      ? `${result.quality.width} × ${result.quality.height} piksel · ${result.quality.issues.length ? 'Perlu perbaikan' : 'Pemeriksaan teknis selesai'}`
                      : 'Gejala didahulukan dari foto'}
                  </dd>
                </div>
                <div>
                  <dt>Validasi klinis</dt>
                  <dd>Belum tersedia</dd>
                </div>
              </dl>
              {result.candidates.length > 0 && (
                <details>
                  <summary>Lihat skor eksperimen</summary>
                  <p className="field-help">
                    Skor relatif antar label dataset. Bukan peluang Anda memiliki penyakit.
                  </p>
                  {result.candidates.map((c) => (
                    <div className="candidate" key={c.code}>
                      <span>{c.name}</span>
                      <strong>{(c.score * 100).toFixed(1)}%</strong>
                      <meter
                        min="0"
                        max="1"
                        value={c.score}
                        aria-label={`Skor eksperimen ${c.name}`}
                      />
                    </div>
                  ))}
                </details>
              )}
              {result.warnings.map((w) => (
                <p className="model-warning" key={w}>
                  <Info size={16} />
                  {w}
                </p>
              ))}
              <h4>Sumber edukasi</h4>
              {result.references.map((r) => (
                <a
                  className="source-link"
                  key={r.url}
                  href={r.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {r.title}
                  <ArrowUpRight size={14} />
                </a>
              ))}
            </div>
          </div>
          <div className="result-actions">
            <button type="button" className="button secondary" onClick={journal}>
              <NotebookPen size={17} />
              Siapkan catatan jurnal
            </button>
            <button
              type="button"
              className="text-button"
              onClick={() => {
                setResult(null);
                document.getElementById('skin-photo')?.focus();
              }}
            >
              <RotateCcw size={16} />
              Tinjau foto & keluhan
            </button>
          </div>
          <p className="privacy-caption">
            <LockKeyhole size={13} />
            {result.privacy}
          </p>
        </section>
      )}
      <Link className="back-link" to="/">
        {' '}
        <ArrowLeft size={16} />
        Kembali ke beranda
      </Link>
    </div>
  );
}
function ScanFrame() {
  return (
    <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <path
        d="M22 8H11a3 3 0 0 0-3 3v11M42 8h11a3 3 0 0 1 3 3v11M56 42v11a3 3 0 0 1-3 3H42M22 56H11a3 3 0 0 1-3-3V42"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <circle cx="32" cy="32" r="13" stroke="currentColor" strokeWidth="1" strokeDasharray="2 4" />
      <path d="M26 32h12m-6-6v12" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}
