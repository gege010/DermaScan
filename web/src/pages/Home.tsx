import {
  ArrowRight,
  ArrowUpRight,
  ScanLine,
  BookOpen,
  NotebookPen,
  LockKeyhole,
  Plus,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import SkinStudy from '../components/SkinStudy';
import { ArrowLink, Art, SectionHead } from '../components/ui';
import articles from '../../../content/education.json';

export default function Home() {
  return (
    <>
      <section className="hero container">
        <div className="hero-copy">
          <p className="eyebrow">
            <span className="status-dot" />
            RUANG UNTUK MEMAHAMI KULITMU
          </p>
          <h1>
            Kenali kulitmu.
            <br />
            Rawat dengan
            <br />
            <span>lebih mengerti.</span>
          </h1>
          <p className="hero-description">
            Dari hal kecil yang kamu rasakan, ke langkah perawatan yang lebih terarah. Pahami
            keluhan kulitmu lewat analisis foto dan edukasi yang terbuka.
          </p>
          <div className="hero-actions">
            <Link className="button primary" to="/analisis">
              Mulai kenali kulit
              <ArrowUpRight size={19} />
            </Link>
            <Link className="quiet-link" to="/pendekatan">
              Cara kami membantu
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
            <span>DI BALIK PERMUKAAN</span>
            <span>STUDI KULIT / 001</span>
          </div>
          <SkinStudy />
          <div className="visual-note">
            <Plus size={15} />
            <span>
              Setiap lapisan punya peran.
              <br />
              Setiap kulit punya cerita.
            </span>
            <span className="visual-note-label">Ilustrasi edukatif</span>
          </div>
        </div>
      </section>
      <div className="manifesto-strip">
        <div className="container">
          <span>Lebih memahami, sebelum merawat.</span>
          <p>
            Tanpa diagnosis instan. Tanpa janji kulit sempurna.
            <br />
            Hanya langkah yang lebih masuk akal untuk kulitmu.
          </p>
          <Link to="/pendekatan" aria-label="Pelajari pendekatan DermaScan">
            <ArrowUpRight size={25} />
          </Link>
        </div>
      </div>
      <section className="container section">
        <SectionHead eyebrow="SATU RUANG, TIGA CARA MERAWAT" title="Mulai dari yang kamu butuhkan.">
          <span className="section-aside">
            Perawatan yang baik dimulai
            <br />
            dengan perhatian.
          </span>
        </SectionHead>
        <div className="path-grid">
          {[
            {
              icon: ScanLine,
              title: 'Pahami yang kamu rasakan',
              body: 'Ceritakan keluhan, tambahkan foto, dan pahami langkah berikutnya beserta batas analisisnya.',
              to: '/analisis',
              label: 'Kenali kulit',
            },
            {
              icon: BookOpen,
              title: 'Temukan penjelasan yang jernih',
              body: 'Baca panduan perawatan yang merujuk sumber dermatologi. Pelan-pelan, satu kebiasaan setiap hari.',
              to: '/panduan',
              label: 'Jelajahi panduan',
            },
            {
              icon: NotebookPen,
              title: 'Perhatikan perubahan kecil',
              body: 'Catat apa yang kulitmu rasakan. Bangun cerita yang lebih lengkap untuk dirimu dan konsultasi nanti.',
              to: '/jurnal',
              label: 'Buka jurnal',
            },
          ].map((p, i) => (
            <article className="path-card" key={p.to}>
              <div className="path-top">
                <p.icon size={26} strokeWidth={1.3} />
                <span>0{i + 1}</span>
              </div>
              <h3>{p.title}</h3>
              <p>{p.body}</p>
              <ArrowLink to={p.to}>{p.label}</ArrowLink>
            </article>
          ))}
        </div>
      </section>
      <section className="care-feature container">
        <div className="care-visual">
          <div className="care-orbit">
            <span className="orbit-core" />
            <span className="orbit-line one" />
            <span className="orbit-line two" />
            <span className="orbit-line three" />
            <span className="orbit-label">
              PERHATIAN
              <br />
              YANG KONSISTEN
            </span>
          </div>
          <span className="small-label">Merawat adalah proses, bukan perlombaan.</span>
        </div>
        <div className="care-copy">
          <p className="eyebrow">ADA CERITA DI SETIAP PERUBAHAN</p>
          <h2>
            Tidak harus langsung tahu.
            <br />
            Mulai dengan mencatat.
          </h2>
          <p>
            Terasa lebih kering setelah ganti pembersih? Muncul benjolan sejak minggu lalu? Hal-hal
            kecil ini membantu kamu memahami pola kulitmu.
          </p>
          <ArrowLink to="/jurnal">Buat catatan pertamamu</ArrowLink>
          <div className="care-footnote">
            <LockKeyhole size={15} />
            <p>
              Catatan disimpan hanya jika kamu memilihnya,
              <br />
              di browser perangkatmu sendiri.
            </p>
          </div>
        </div>
      </section>
      <section className="container section">
        <SectionHead eyebrow="CATATAN UNTUK KULITMU" title="Sedikit belajar. Lebih mengenal.">
          <ArrowLink to="/panduan">Semua panduan</ArrowLink>
        </SectionHead>
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
      <section className="bottom-invitation container">
        <span className="eyebrow">LANGKAH KECIL, MULAI HARI INI</span>
        <h2>
          Berikan kulitmu
          <br />
          sedikit lebih banyak perhatian.
        </h2>
        <Link className="button primary" to="/analisis">
          Mari mulai
          <ArrowUpRight size={18} />
        </Link>
      </section>
    </>
  );
}
