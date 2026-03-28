"""
utils/tavily_search.py
─────────────────────────────────────────────────────────────────────────────
Modul (OOP) untuk melakukan pencarian referensi web menggunakan Tavily API.
Mengambil artikel medis, jurnal, atau produk perawatan kulit terkait.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from tavily import TavilyClient
from utils.logger import get_logger

logger = get_logger(__name__)

class TavilySearch:
    def __init__(self):
        """Inisialisasi koneksi ke Tavily saat server FastAPI pertama kali menyala."""
        self.api_key = os.getenv("TAVILY_API_KEY")
        if self.api_key:
            self.client = TavilyClient(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("⚠️ TAVILY_API_KEY belum dikonfigurasi di .env")

    def search(self, query: str) -> dict:
        """
        Melakukan pencarian cerdas di web berdasarkan nama kondisi kulit.
        """
        if not self.client:
            return {
                "query": query,
                "ringkasan_ai": "Pencarian web dinonaktifkan karena TAVILY_API_KEY belum diatur.",
                "hasil_pencarian": []
            }

        try:
            # Mencari dengan mode "advanced" dan menyertakan ringkasan (answer)
            response = self.client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
                max_results=3
            )
            
            # Memformat hasil agar sesuai dengan apa yang diharapkan oleh antarmuka Streamlit
            return {
                "ringkasan_ai": response.get("answer", "Ringkasan otomatis tidak tersedia."),
                "hasil_pencarian": [
                    {
                        "title": res.get("title", "Tautan Tanpa Judul"), 
                        "url": res.get("url", "#"), 
                        "snippet": res.get("content", "")
                    }
                    for res in response.get("results", [])
                ]
            }

        except Exception as e:
            logger.error(f"Tavily API error: {e}")
            return {
                "error_tavily": f"Terjadi kesalahan saat mencari di web: {str(e)}",
                "hasil_pencarian": []
            }