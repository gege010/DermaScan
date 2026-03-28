"""
utils/groq_analyzer.py
─────────────────────────────────────────────────────────────────────────────
Modul (OOP) untuk mengirimkan hasil prediksi CNN ke Groq API (Llama 3)
guna mendapatkan interpretasi medis awam yang sangat detail dan terstruktur.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
from groq import Groq
from utils.logger import get_logger

logger = get_logger(__name__)

class GroqAnalyzer:
    def __init__(self):
        """Inisialisasi koneksi ke Groq saat server FastAPI pertama kali menyala."""
        self.api_key = os.getenv("GROQ_API_KEY")
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("⚠️ GROQ_API_KEY belum dikonfigurasi di .env")

    def analyze(self, condition_name: str, confidence: float) -> dict:
        """Mengirimkan nama kondisi kulit ke Groq dan meminta penjelasan rinci."""
        if not self.client:
            return {
                "error_groq": "GROQ_API_KEY belum dikonfigurasi di .env",
                "penjelasan_singkat": f"Terdeteksi kondisi: {condition_name}.",
                "rekomendasi_perawatan": ["Silakan konsultasi ke dokter spesialis kulit."],
                "bahan_aktif_disarankan": ["Pembersih wajah yang lembut."]
            }

        # prompt telah dibersihkan dari query_pencarian_produk agar sinkron dengan app.py
        system_prompt = f"""
Anda adalah seorang asisten dermatologis AI ahli.
Tugas Anda adalah memberikan penjelasan edukatif kepada pengguna awam mengenai kondisi kulit yang terdeteksi oleh sistem kami.
Kondisi yang terdeteksi: '{condition_name}' (Tingkat Keyakinan AI: {confidence:.1%}).

ANDA WAJIB MENGEMBALIKAN OUTPUT MURNI DALAM FORMAT JSON DENGAN STRUKTUR BERIKUT:
{{
  "penjelasan_singkat": "Paragraf 3-4 kalimat yang menjelaskan apa itu kondisi ini secara mendalam namun mudah dipahami, apa penyebab utamanya, dan apakah ini menular atau berbahaya.",
  "rekomendasi_perawatan": [
    "Langkah perawatan spesifik 1 secara rinci...",
    "Langkah perawatan spesifik 2 secara rinci...",
    "Langkah perawatan spesifik 3 secara rinci...",
    "Hal yang dilarang keras untuk dilakukan (misal: memencet, memakai scrub kasar, dll)"
  ],
  "bahan_aktif_disarankan": [
    "Nama Bahan 1 (Penjelasan singkat fungsinya)",
    "Nama Bahan 2 (Penjelasan singkat fungsinya)",
    "Nama Bahan 3 (Penjelasan singkat fungsinya)"
  ]
}}

Catatan Penting:
- JSON harus Valid. Jangan ada teks markdown seperti ```json di awal atau akhir.
- Gunakan bahasa Indonesia yang profesional dan empatik.
- Isi array rekomendasi_perawatan dan bahan_aktif_disarankan minimal masing-masing 3-4 poin yang sangat mendetail!
"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            result_content = response.choices[0].message.content
            return json.loads(result_content)

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "error_groq": f"Terjadi kesalahan saat menghubungi Groq: {str(e)}",
                "penjelasan_singkat": f"Terdeteksi kondisi: {condition_name}.",
                "rekomendasi_perawatan": ["Silakan konsultasi ke dokter spesialis kulit."],
                "bahan_aktif_disarankan": ["Pembersih wajah yang lembut."]
            }