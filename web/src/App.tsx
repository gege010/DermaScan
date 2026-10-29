import { useEffect, useState } from 'react';
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { ArrowUpRight, Menu, X } from 'lucide-react';
import { Brand } from './components/ui';
import { SessionProvider } from './session';
import Home from './pages/Home';
import Scan from './pages/Scan';
import Learn, { Article } from './pages/Learn';
import Journal from './pages/Journal';
import Method from './pages/Method';

const links = [
  { to: '/analisis', label: 'Kenali kulit' },
  { to: '/panduan', label: 'Ruang belajar' },
  { to: '/jurnal', label: 'Jurnal kulit' },
  { to: '/pendekatan', label: 'Pendekatan kami' },
];
function Shell() {
  const [menu, setMenu] = useState(false);
  const location = useLocation();
  useEffect(() => {
    setMenu(false);
    window.scrollTo({ top: 0, behavior: 'instant' });
    const page = links.find((l) => location.pathname.startsWith(l.to));
    document.title = `${page?.label || 'Ruang untuk memahami kulitmu'} — DermaScan`;
    document.getElementById('main')?.focus({ preventScroll: true });
  }, [location.pathname]);
  return (
    <>
      <a className="skip-link" href="#main">
        Lewati ke konten
      </a>
      <header className="site-header">
        <div className="header-inner">
          <Brand />
          <nav className="desktop-nav" aria-label="Navigasi utama">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to}>
                {l.label}
              </NavLink>
            ))}
          </nav>
          <Link className="header-cta" to="/analisis">
            Mulai analisis
            <ArrowUpRight size={16} />
          </Link>
          <button
            className="icon-button menu-toggle"
            type="button"
            aria-label={menu ? 'Tutup navigasi' : 'Buka navigasi'}
            aria-expanded={menu}
            aria-controls="mobile-nav"
            onClick={() => setMenu(!menu)}
          >
            {menu ? <X /> : <Menu />}
          </button>
        </div>
        {menu && (
          <nav id="mobile-nav" className="mobile-nav" aria-label="Navigasi seluler">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to}>
                {l.label}
                <ArrowUpRight size={18} />
              </NavLink>
            ))}
          </nav>
        )}
      </header>
      <main id="main" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analisis" element={<Scan />} />
          <Route path="/panduan" element={<Learn />} />
          <Route path="/panduan/:id" element={<Article />} />
          <Route path="/jurnal" element={<Journal />} />
          <Route path="/pendekatan" element={<Method />} />
          <Route
            path="*"
            element={
              <div className="page container">
                <p className="eyebrow">Halaman tidak ditemukan</p>
                <h1>Kembali ke ruangmu.</h1>
                <Link className="button primary" to="/">
                  Ke beranda
                  <ArrowUpRight size={18} />
                </Link>
              </div>
            }
          />
        </Routes>
      </main>
      <footer className="site-footer container">
        <div className="footer-top">
          <Brand />
          <p>
            Kulit lebih dipahami.
            <br />
            Perawatan lebih disadari.
          </p>
          <Link to="/pendekatan">
            Kemampuan & privasi
            <ArrowUpRight size={16} />
          </Link>
        </div>
        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} DermaScan</span>
          <p>Teman edukasi kulit. Bukan pengganti diagnosis atau pemeriksaan dokter.</p>
          <span>Dibuat dengan perhatian.</span>
        </div>
      </footer>
    </>
  );
}
export default function App() {
  return (
    <SessionProvider>
      <Shell />
    </SessionProvider>
  );
}
