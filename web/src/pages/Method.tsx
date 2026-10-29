import { ArrowUpRight, Eye, Fingerprint, Layers3, LockKeyhole } from 'lucide-react';
import { Link } from 'react-router-dom';
import SkinStudy from '../components/SkinStudy';

export default function Method() {
  return (
    <div className="container page method-page">
      <div className="method-hero">
        <div className="page-intro">
          <p className="eyebrow">PENDEKATAN KAMI</p>
          <h1>
            Teknologi yang
            <br />
            tahu batasnya.
          </h1>
          <p>
            Perawatan kulit membutuhkan konteks, perhatian,
            <br />
            dan penilaian manusia. Kami mulai dari sana.
          </p>
        </div>
        <SkinStudy compact />
      </div>
      <div className="method-statement">
        <span className="eyebrow">PRINSIP DERMASCAN</span>
        <h2>
          Lebih baik jujur tentang yang belum diketahui,
          <br />
          daripada meyakinkan tanpa dasar.
        </h2>
      </div>
      <div className="method-grid">
        {[
          {
            Icon: Eye,
            title: 'Apa yang bisa dibantu',
            body: 'Mencatat keluhan, memeriksa kualitas teknis foto, mengenali gejala yang perlu ditindaklanjuti, serta menyediakan edukasi dengan sumber yang dapat dibaca.',
          },
          {
            Icon: Layers3,
            title: 'Apa yang belum bisa dipastikan',
            body: 'Diagnosis, tingkat keparahan penyakit, keamanan suatu obat, atau apakah sebuah lesi jinak. Foto biasa dan dermoskopi juga memiliki cakupan yang berbeda.',
          },
          {
            Icon: Fingerprint,
            title: 'Tentang model dan skor',
            body: 'Model lokal bersifat eksperimen dan belum tervalidasi klinis. Jika diaktifkan operator, skor hanya menunjukkan kemiripan dengan label dataset. Skor tinggi bukan jaminan kebenaran.',
          },
          {
            Icon: LockKeyhole,
            title: 'Foto dan catatanmu',
            body: 'Aplikasi tidak menyimpan foto atau mengirimkannya ke penyedia AI eksternal. Foto diproses sementara di server. Jurnal disimpan di browser hanya ketika kamu mengaktifkan penyimpanan perangkat.',
          },
        ].map(({ Icon, title, body }) => (
          <section key={title}>
            <Icon size={25} strokeWidth={1.3} />
            <h2>{title}</h2>
            <p>{body}</p>
          </section>
        ))}
      </div>
      <section className="method-detail">
        <div>
          <p className="eyebrow">TRANSPARANSI, BUKAN JANJI</p>
          <h2>
            Dirancang untuk membantu.
            <br />
            Belum untuk mendiagnosis.
          </h2>
        </div>
        <div>
          <p>
            DermaScan berasal dari proyek machine learning klasifikasi kondisi kulit. Versi ini
            memperbaiki cara hasil disampaikan: tidak memaksakan diagnosis, tidak menampilkan
            akurasi penelitian sebagai jaminan untuk foto baru, dan tidak memberikan resep otomatis.
          </p>
          <p>
            Materi edukasi merujuk sumber publik seperti American Academy of Dermatology dan NHS.
            Ringkasan di aplikasi belum ditinjau klinisi DermaScan. Sebelum penggunaan sebagai
            layanan medis, diperlukan validasi eksternal, tinjauan klinis, serta penilaian keamanan
            dan regulasi yang sesuai.
          </p>
          <p>
            Jurnal tidak terenkripsi secara terpisah dari browser dan tidak memiliki pencadangan.
            Pengguna perangkat yang sama dapat mengaksesnya. Kamu dapat mengekspor catatan,
            menghapusnya satu per satu, atau menonaktifkan penyimpanan untuk menghapus salinan di
            browser.
          </p>
          <a
            className="arrow-link"
            href="https://github.com/gege010/DermaScan"
            target="_blank"
            rel="noreferrer"
          >
            Lihat proyek terbuka
            <ArrowUpRight size={17} />
          </a>
        </div>
      </section>
      <section className="emergency-panel">
        <p className="eyebrow">SAAT TIDAK BISA MENUNGGU</p>
        <h2>
          Gejala serius membutuhkan
          <br />
          pertolongan langsung.
        </h2>
        <p>
          Sulit bernapas, pembengkakan wajah, atau kulit melepuh dengan demam atau luka pada mulut /
          mata? Cari pertolongan medis segera. Jangan menunggu hasil aplikasi.
        </p>
        <a
          className="arrow-link"
          href="https://www.nhs.uk/conditions/stevens-johnson-syndrome/"
          target="_blank"
          rel="noreferrer"
        >
          Baca tanda reaksi serius di NHS
          <ArrowUpRight size={17} />
        </a>
      </section>
      <Link className="button primary" to="/analisis">
        Mulai dengan memahami
        <ArrowUpRight size={17} />
      </Link>
    </div>
  );
}
