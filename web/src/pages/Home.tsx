import {
  ArrowRight,
  ArrowUpRight,
  ScanLine,
  BookOpen,
  NotebookPen,
  LockKeyhole,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import SkinStudy from '../components/SkinStudy';
import { ArrowLink, Art } from '../components/ui';
import articles from '../../../content/education.json';

export default function Home() {
  return (
    <>
      <section className="hero container">
        <div className="hero-copy">
          <h1>
            Kenali kulitmu <em>lebih dekat.</em>
          </h1>
          <p className="hero-description">
            Pahami keluhan kulit lewat foto dan gejala, baca panduan bersumber, dan catat perubahan
            kulitmu secara pribadi.
          </p>
          <div className="hero-actions">
            <Link className="button primary" to="/analisis">
              Mulai kenali kulit
              <ArrowUpRight size={19} />
            </Link>
            <Link className="quiet-link" to="/panduan">
              Lihat panduan
              <ArrowRight size={16} />
            </Link>
          </div>
          <p className="privacy-caption">
            <LockKeyhole size={13} />
            Foto diproses tanpa disimpan oleh aplikasi.
          </p>
        </div>
        <div className="hero-visual">
          <div className="visual-caption">
            <span>ATLAS KULIT</span>
            <span>Ilustrasi edukatif</span>
          </div>
          <SkinStudy />
        </div>
      </section>
      <section className="container section home-directory" aria-labelledby="directory-title">
        <div className="section-head">
          <h2 id="directory-title">Apa yang ingin kamu lakukan?</h2>
        </div>
        <div className="path-grid">
          {[
            {
              icon: ScanLine,
              title: 'Analisis keluhan',
              body: 'Ceritakan gejalamu. Tambahkan foto jika perlu, lalu lihat langkah berikutnya.',
              to: '/analisis',
            },
            {
              icon: BookOpen,
              title: 'Baca panduan',
              body: 'Temukan panduan perawatan kulit dan telusuri sumber dermatologinya.',
              to: '/panduan',
            },
            {
              icon: NotebookPen,
              title: 'Catat perubahan',
              body: 'Simpan catatan gejala dan perubahan kulit untuk konsultasi berikutnya.',
              to: '/jurnal',
            },
          ].map((item) => (
            <Link className="path-card" key={item.to} to={item.to}>
              <item.icon size={22} strokeWidth={1.5} />
              <div>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </div>
              <ArrowUpRight className="path-arrow" size={20} aria-hidden="true" />
            </Link>
          ))}
        </div>
      </section>
      <section className="container section home-reading" aria-labelledby="reading-title">
        <div className="section-head">
          <h2 id="reading-title">Panduan pilihan</h2>
          <ArrowLink to="/panduan">Semua panduan</ArrowLink>
        </div>
        <div className="article-grid">
          {articles.slice(0, 3).map((a) => (
            <Link className="article-card" to={`/panduan/${a.id}`} key={a.id}>
              <Art kind={a.art} />
              <div className="article-meta">
                <span>{a.category}</span>
                <span>{a.minutes} menit baca</span>
              </div>
              <h3>
                {a.title}
                <ArrowUpRight size={20} />
              </h3>
              <p>{a.excerpt}</p>
            </Link>
          ))}
        </div>
      </section>
    </>
  );
}
