import { useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ArrowLeft, ArrowUpRight, BookOpen, Search, X } from 'lucide-react';
import { Art } from '../components/ui';
import articles from '../../../content/education.json';

export default function Learn() {
  const [params, setParams] = useSearchParams();
  const query = params.get('q') || '';
  const category = params.get('kategori') || 'Semua';
  const [composing, setComposing] = useState(false);
  const categories = ['Semua', ...new Set(articles.map((a) => a.category))];
  const filtered = articles.filter(
    (a) =>
      (category === 'Semua' || a.category === category) &&
      `${a.title} ${a.excerpt}`
        .toLocaleLowerCase('id-ID')
        .includes(query.toLocaleLowerCase('id-ID')),
  );
  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value && value !== 'Semua') next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  }
  return (
    <div className="container page learn-page">
      <div className="page-intro">
        <p className="eyebrow">RUANG BELAJAR</p>
        <h1>
          Kenal lebih dekat.
          <br />
          Rawat lebih sadar.
        </h1>
        <p>
          Penjelasan sederhana, sumber yang bisa kamu telusuri.
          <br />
          Untuk keputusan kecil yang berarti bagi kulitmu.
        </p>
      </div>
      <div className="library-tools">
        <div className="filter-buttons" aria-label="Kategori panduan">
          {categories.map((c) => (
            <button
              type="button"
              key={c}
              aria-pressed={category === c}
              onClick={() => update('kategori', c)}
            >
              {c}
            </button>
          ))}
        </div>
        <div className="search-field">
          <Search size={17} />
          <label className="visually-hidden" htmlFor="article-search">
            Cari panduan
          </label>
          <input
            id="article-search"
            type="search"
            value={query}
            onCompositionStart={() => setComposing(true)}
            onCompositionEnd={() => setComposing(false)}
            onChange={(e) => update('q', e.target.value)}
            onKeyDown={(e) => {
              if (composing && e.key === 'Enter') e.preventDefault();
            }}
            placeholder="Cari panduan…"
          />
          {query && (
            <button
              type="button"
              aria-label="Hapus pencarian"
              className="icon-button"
              onClick={() => {
                update('q', '');
                document.getElementById('article-search')?.focus();
              }}
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>
      <p className="result-count" role="status">
        {filtered.length} panduan untuk kamu
      </p>
      {filtered.length ? (
        <div className="article-grid library-grid">
          {filtered.map((a) => (
            <Link className="article-card" to={`/panduan/${a.id}`} key={a.id}>
              <Art kind={a.art} />
              <div className="article-meta">
                <span>{a.category}</span>
                <span>{a.minutes} menit baca</span>
              </div>
              <h2>
                {a.title}
                <ArrowUpRight size={19} />
              </h2>
              <p>{a.excerpt}</p>
            </Link>
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <BookOpen size={32} strokeWidth={1} />
          <h2>Belum menemukan yang kamu cari.</h2>
          <p>Coba kata yang lebih umum atau lihat seluruh panduan.</p>
          <button type="button" className="button secondary" onClick={() => setParams({})}>
            Lihat semua panduan
            <ArrowUpRight size={17} />
          </button>
        </div>
      )}
      <div className="editorial-note">
        <span className="eyebrow">TENTANG PANDUAN INI</span>
        <p>
          Dirangkum dari sumber dermatologi publik. Materi edukatif untuk orang dewasa, belum
          ditinjau klinisi DermaScan. Kebutuhan setiap orang dapat berbeda.
        </p>
      </div>
    </div>
  );
}
export function Article() {
  const { id } = useParams();
  const article = articles.find((a) => a.id === id);
  if (!article)
    return (
      <div className="container page empty-state">
        <h1>Panduan tidak ditemukan.</h1>
        <Link className="button primary" to="/panduan">
          Kembali ke ruang belajar
        </Link>
      </div>
    );
  return (
    <article className="container page article-page">
      <Link className="back-link" to="/panduan">
        <ArrowLeft size={16} />
        Ruang belajar
      </Link>
      <div className="article-header">
        <p className="eyebrow">
          {article.category} · {article.minutes} MENIT BACA
        </p>
        <h1>{article.title}</h1>
        <p>{article.excerpt}</p>
      </div>
      <Art kind={article.art} />
      <div className="article-body">
        {article.sections.map((s) => (
          <section key={s.title}>
            <h2>{s.title}</h2>
            <p>{s.body}</p>
          </section>
        ))}
        <div className="article-source">
          <p className="eyebrow">BACA SUMBER ASLINYA</p>
          <a href={article.source.url} target="_blank" rel="noreferrer">
            {article.source.title}
            <ArrowUpRight size={17} />
          </a>
          <p>
            Ringkasan edukatif, bukan panduan pengobatan pribadi. Belum ditinjau klinisi DermaScan.
          </p>
        </div>
        <div className="article-next">
          <h3>Ingin mencatat yang kamu rasakan?</h3>
          <Link className="arrow-link" to="/jurnal">
            Buka jurnal kulit
            <ArrowUpRight size={17} />
          </Link>
        </div>
      </div>
    </article>
  );
}
