"""Conservative educational policy; not a clinically validated triage engine."""

SOURCES = [
    {
        "title": "Dasar perawatan kulit · AAD",
        "url": "https://www.aad.org/public/everyday-care/skin-care-basics",
    },
    {
        "title": "Tanda reaksi kulit serius · NHS",
        "url": "https://www.nhs.uk/conditions/stevens-johnson-syndrome/",
    },
    {
        "title": "Mengenali perubahan lesi · AAD",
        "url": "https://www.aad.org/public/diseases/skin-cancer",
    },
]
SYMPTOMS = {
    "itch": "Gatal",
    "pain": "Nyeri",
    "dryness": "Kering atau mengelupas",
    "redness": "Kemerahan",
    "breakouts": "Jerawat atau benjolan",
    "changing_mole": "Tahi lalat berubah atau luka tidak sembuh",
    "breathing": "Sulit bernapas atau wajah membengkak",
    "blisters_fever": "Kulit melepuh disertai demam atau luka di mulut/mata",
}
EMERGENCY = {"breathing", "blisters_fever"}
DISPLAY_NAMES = {
    "acne": "Jerawat",
    "dry": "Kulit kering",
    "oily": "Kulit berminyak",
    "normal": "Kulit tanpa kelainan pada label dataset",
    "combination": "Kulit kombinasi",
    "hyperpigmentation": "Hiperpigmentasi",
    "Nail_psoriasis": "Psoriasis kuku",
    "SJS-TEN": "SJS/TEN",
    "Vitiligo": "Vitiligo",
    "akiec": "Keratosis aktinik",
    "bcc": "Karsinoma sel basal",
    "bkl": "Lesi keratosis jinak",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanositik",
    "vasc": "Lesi vaskular",
}
REVIEW_CLASSES = {"mel", "bcc", "akiec", "SJS-TEN"}


def initial_assessment(symptoms: list[str]) -> dict:
    if EMERGENCY.intersection(symptoms):
        return {
            "status": "urgent",
            "title": "Utamakan pertolongan medis sekarang",
            "summary": "Gejala yang Anda pilih dapat memerlukan penanganan segera. Jangan menunggu hasil foto untuk mencari pertolongan di IGD terdekat.",
            "next_steps": [
                "Minta bantuan orang terdekat untuk mencari pertolongan medis.",
                "Bawa daftar obat dan waktu mulai munculnya keluhan.",
            ],
        }
    if "changing_mole" in symptoms:
        return {
            "status": "review",
            "title": "Perubahan ini perlu diperiksa langsung",
            "summary": "Foto saja tidak dapat memastikan penyebab tahi lalat yang berubah atau luka yang tidak sembuh. Atur pemeriksaan dengan dokter kulit.",
            "next_steps": [
                "Catat sejak kapan perubahan muncul dan apakah membesar atau berdarah.",
                "Bawa foto sebelumnya bila ada. Jangan mengandalkan skor AI untuk menunda pemeriksaan.",
            ],
        }
    return {
        "status": "unsupported",
        "title": "Foto belum dapat diberi kesimpulan medis",
        "summary": "DermaScan dapat membantu menyiapkan catatan keluhan dan edukasi. Model foto pada proyek ini belum tervalidasi untuk diagnosis pada penggunaan publik.",
        "next_steps": [
            "Catat perubahan, pemicu, dan produk yang baru digunakan.",
            "Jika keluhan menetap, memburuk, atau terasa mengganggu, periksakan ke dokter.",
        ],
    }


def education_for(symptoms: list[str]) -> list[dict]:
    items = [
        {
            "title": "Mulai dari perawatan sederhana",
            "body": "Gunakan pembersih lembut dan hindari menggosok kulit. Hentikan produk baru yang terasa mengiritasi. Panduan ini bukan resep untuk kondisi tertentu.",
        },
        {
            "title": "Lindungi kulit dari matahari",
            "body": "Cari tempat teduh dan gunakan pelindung matahari sesuai petunjuk produk. AAD menyarankan perlindungan spektrum luas SPF 30 atau lebih untuk rutinitas dewasa.",
        },
    ]
    if "dryness" in symptoms:
        items.insert(
            0,
            {
                "title": "Bantu menjaga kelembapan",
                "body": "Pilih pelembap tanpa pewangi dan hindari air yang terlalu panas. Keluhan kering dapat memiliki banyak penyebab; foto tidak cukup untuk membedakannya.",
            },
        )
    if "breakouts" in symptoms:
        items.insert(
            0,
            {
                "title": "Rawat tanpa memencet",
                "body": "Hindari memencet benjolan dan pertimbangkan produk berlabel nonkomedogenik. Jerawat yang nyeri, meninggalkan bekas, atau menetap perlu dibahas dengan dokter.",
            },
        )
    return items
