import { useRef, useState } from 'react';
import { ArrowDownToLine, Check, LockKeyhole, NotebookPen, Plus, Trash2 } from 'lucide-react';
import { Dialog, Notice } from '../components/ui';
import { useSession } from '../session';
import type { JournalEntry } from '../types';
import { JOURNAL_KEY as KEY } from '../storage';

export default function Journal() {
  const {
    journalDraft,
    setJournalDraft,
    entries,
    setEntries,
    persistent,
    setPersistent,
    journalEditor,
    setJournalEditor,
  } = useSession();
  const { feeling, area, editing } = journalEditor;
  const setFeeling = (value: string) => setJournalEditor((d) => ({ ...d, feeling: value }));
  const setArea = (value: string) => setJournalEditor((d) => ({ ...d, area: value }));
  const setEditing = (value: string | null) => setJournalEditor((d) => ({ ...d, editing: value }));
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [remove, setRemove] = useState<JournalEntry | null>(null);
  const [disableStorage, setDisableStorage] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  function commit(next: JournalEntry[]) {
    try {
      if (persistent) localStorage.setItem(KEY, JSON.stringify(next));
      setEntries(next);
      return true;
    } catch {
      setError(
        'Penyimpanan browser penuh atau diblokir. Catatan belum berubah. Nonaktifkan penyimpanan perangkat atau ekspor catatan sebelumnya.',
      );
      return false;
    }
  }
  function save(e: React.SubmitEvent) {
    e.preventDefault();
    setError('');
    if (!journalDraft.trim()) {
      setError('Tulis hal yang kamu rasakan sebelum menyimpan.');
      textarea.current?.focus();
      return;
    }
    if (!editing && entries.length >= 100) {
      setError('Jurnal maksimal 100 catatan. Ekspor dan hapus catatan lama sebelum menambah.');
      return;
    }
    const original = entries.find((n) => n.id === editing);
    if (editing && !original) {
      setEditing(null);
      setError(
        'Catatan telah dihapus di tab lain. Teks tetap tersedia; simpan sebagai catatan baru.',
      );
      return;
    }
    const entry = {
      id: editing || crypto.randomUUID(),
      date: original?.date || new Date().toISOString(),
      area,
      feeling,
      note: journalDraft.trim(),
    };
    if (
      commit(editing ? entries.map((n) => (n.id === editing ? entry : n)) : [entry, ...entries])
    ) {
      setJournalDraft('');
      setEditing(null);
      setMessage(
        persistent
          ? 'Catatan tersimpan di browser perangkat ini.'
          : 'Catatan ditambahkan untuk sesi ini. Aktifkan penyimpanan perangkat agar tersedia setelah halaman dimuat ulang.',
      );
    }
  }
  function toggleStorage() {
    if (persistent) {
      setDisableStorage(true);
      return;
    }
    try {
      localStorage.setItem(KEY, JSON.stringify(entries));
      setPersistent(true);
      setMessage('Penyimpanan perangkat aktif. Catatan dapat dibaca pengguna lain di browser ini.');
    } catch {
      setError('Penyimpanan browser tidak tersedia. Catatan tetap tersedia dalam sesi ini.');
    }
  }
  function download() {
    const body = entries
      .map(
        (n) => `${new Date(n.date).toLocaleString('id-ID')} | ${n.area} | ${n.feeling}\n${n.note}`,
      )
      .join('\n\n────────\n\n');
    const url = URL.createObjectURL(
      new Blob([`JURNAL KULIT — DERMASCAN\nCatatan pribadi; bukan diagnosis.\n\n${body}`], {
        type: 'text/plain;charset=utf-8',
      }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'jurnal-kulit.txt';
    a.click();
    URL.revokeObjectURL(url);
    setMessage('Jurnal diekspor. Simpan file yang berisi catatan pribadi ini dengan hati-hati.');
  }
  return (
    <div className="container page journal-page">
      <div className="page-intro">
        <p className="eyebrow">JURNAL KULIT</p>
        <h1>
          Perubahan kecil,
          <br />
          cerita yang berarti.
        </h1>
        <p>
          Ruang pribadi untuk mencatat apa yang kulitmu rasakan.
          <br />
          Bukan untuk mengejar sempurna. Untuk lebih memahami.
        </p>
      </div>
      <div className="journal-layout">
        <section className="journal-editor">
          <div className="editor-title">
            <NotebookPen size={23} strokeWidth={1.3} />
            <h2>{editing ? 'Sunting catatan' : 'Bagaimana kulitmu hari ini?'}</h2>
          </div>
          <form noValidate onSubmit={save}>
            <fieldset className="choice-fieldset">
              <legend>Area yang diperhatikan</legend>
              <div className="radio-options">
                {['Wajah', 'Tubuh', 'Tangan / kaki', 'Lainnya'].map((v) => (
                  <label className={`choice ${area === v ? 'selected' : ''}`} key={v}>
                    <input
                      type="radio"
                      name="journal-area"
                      checked={area === v}
                      onChange={() => setArea(v)}
                    />
                    {v}
                  </label>
                ))}
              </div>
            </fieldset>
            <fieldset className="choice-fieldset">
              <legend>Dibanding sebelumnya</legend>
              <div className="radio-options">
                {['Lebih nyaman', 'Kurang lebih sama', 'Lebih mengganggu', 'Belum yakin'].map(
                  (v) => (
                    <label className={`choice ${feeling === v ? 'selected' : ''}`} key={v}>
                      <input
                        type="radio"
                        name="feeling"
                        checked={feeling === v}
                        onChange={() => setFeeling(v)}
                      />
                      {v}
                    </label>
                  ),
                )}
              </div>
            </fieldset>
            <label className="field-label" htmlFor="journal-note">
              Hal yang ingin kamu ingat
            </label>
            <textarea
              ref={textarea}
              id="journal-note"
              maxLength={2000}
              rows={6}
              className="journal-textarea resize-none"
              value={journalDraft}
              style={{ resize: 'none' }}
              aria-invalid={!!error}
              aria-describedby="journal-error note-help"
              onChange={(e) => {
                setJournalDraft(e.target.value);
                setError('');
              }}
              placeholder="Misalnya: pipi terasa lebih kering sejak ganti pembersih tiga hari lalu. Tidak gatal, tapi terasa tertarik setelah mencuci wajah."
            />
            <div className="textarea-caption">
              <span id="note-help">Keluhan, pemicu, produk baru, atau perubahan.</span>
              <span>{journalDraft.length}/2000</span>
            </div>
            <p className="field-error" id="journal-error" role={error ? 'alert' : undefined}>
              {error}
            </p>
            <button type="submit" className="button primary">
              <Plus size={17} />
              {editing ? 'Simpan perubahan' : 'Tambahkan catatan'}
            </button>
            {editing && (
              <button
                type="button"
                className="text-button cancel-edit"
                onClick={() => {
                  setEditing(null);
                  setJournalDraft('');
                }}
              >
                Batal menyunting
              </button>
            )}
          </form>
          <div className="storage-choice">
            <LockKeyhole size={18} />
            <div>
              <label className="check-row">
                <input type="checkbox" checked={persistent} onChange={toggleStorage} />
                <strong>Simpan di perangkat ini</strong>
              </label>
              <p>
                Opsional. Catatan tersimpan di browser, tanpa akun dan tanpa sinkronisasi. Hindari
                perangkat bersama. Jika nonaktif, catatan baru hilang setelah halaman dimuat ulang.
              </p>
            </div>
          </div>
        </section>
        <section className="journal-entries">
          <div className="entries-heading">
            <h2>
              Catatanmu <span>{entries.length}</span>
            </h2>
            <button
              type="button"
              className="text-button"
              disabled={!entries.length}
              onClick={download}
            >
              <ArrowDownToLine size={15} />
              Ekspor
            </button>
          </div>
          <p className="journal-status" role="status">
            {message}
          </p>
          {!entries.length ? (
            <div className="journal-empty">
              <div className="notebook-art" aria-hidden="true">
                <span />
                <span />
                <span />
                <span />
              </div>
              <h3>
                Cerita pertamamu
                <br />
                dimulai di sini.
              </h3>
              <p>
                Catat satu hal yang kamu rasakan hari ini.
                <br />
                Besok, kamu punya sesuatu untuk dibandingkan.
              </p>
            </div>
          ) : (
            entries.map((n) => (
              <article className="journal-entry" key={n.id}>
                <div className="entry-meta">
                  <time dateTime={n.date}>
                    {new Date(n.date).toLocaleDateString('id-ID', {
                      day: 'numeric',
                      month: 'long',
                      year: 'numeric',
                    })}
                  </time>
                  <span>{n.area}</span>
                </div>
                <span className="feeling-label">{n.feeling}</span>
                <p>{n.note}</p>
                <div className="entry-actions">
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => {
                      setEditing(n.id);
                      setJournalDraft(n.note);
                      setArea(n.area);
                      setFeeling(n.feeling);
                      textarea.current?.focus();
                    }}
                  >
                    Sunting
                  </button>
                  <button
                    type="button"
                    className="text-button danger-text"
                    onClick={() => setRemove(n)}
                    aria-label={`Hapus catatan ${new Date(n.date).toLocaleDateString('id-ID')}`}
                  >
                    <Trash2 size={14} />
                    Hapus
                  </button>
                </div>
              </article>
            ))
          )}
          <Notice>
            <LockKeyhole size={16} />
            Jurnal ini tidak dipantau tenaga medis. Jika keluhan memburuk, cari bantuan langsung.
          </Notice>
        </section>
      </div>
      <Dialog open={!!remove} title="Hapus catatan ini?" onClose={() => setRemove(null)}>
        <p>
          Catatan akan dihapus dari jurnal{persistent ? ' dan browser perangkat ini' : ''}. Tindakan
          ini tidak dapat dibatalkan.
        </p>
        <div className="dialog-actions">
          <button
            type="button"
            className="button secondary"
            autoFocus
            onClick={() => setRemove(null)}
          >
            Simpan catatan
          </button>
          <button
            type="button"
            className="button danger"
            onClick={() => {
              if (commit(entries.filter((n) => n.id !== remove?.id))) {
                if (editing === remove?.id) {
                  setEditing(null);
                  setJournalDraft('');
                }
                setRemove(null);
                setMessage('Catatan dihapus.');
              }
            }}
          >
            Hapus catatan
          </button>
        </div>
      </Dialog>
      <Dialog
        open={disableStorage}
        title="Nonaktifkan penyimpanan perangkat?"
        onClose={() => setDisableStorage(false)}
      >
        <p>
          Salinan jurnal di browser ini akan dihapus. Catatan tetap tersedia selama sesi ini dan
          bisa kamu ekspor terlebih dahulu.
        </p>
        <div className="dialog-actions">
          <button
            type="button"
            className="button secondary"
            autoFocus
            onClick={() => setDisableStorage(false)}
          >
            Tetap simpan
          </button>
          <button
            type="button"
            className="button danger"
            onClick={() => {
              try {
                localStorage.removeItem(KEY);
                setPersistent(false);
                setDisableStorage(false);
                setMessage(
                  'Penyimpanan perangkat nonaktif. Catatan tersedia hanya untuk sesi ini.',
                );
              } catch {
                setError('Pengaturan browser tidak dapat diubah. Coba lagi.');
              }
            }}
          >
            <Check size={16} />
            Nonaktifkan
          </button>
        </div>
      </Dialog>
    </div>
  );
}
