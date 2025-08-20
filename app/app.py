import torch
import librosa
from transformers import pipeline
from pydub import AudioSegment
import os
import time
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import jellyfish
# Tambahkan jsonify di sini
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import uuid # Untuk nama file unik

# ==============================================================================
# KONFIGURASI APLIKASI FLASK
# ==============================================================================
# Path sekarang merujuk ke luar folder app
UPLOAD_FOLDER = '../uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm'} # Tambahkan webm untuk rekaman browser

app = Flask(__name__)
# Sesuaikan path ke folder upload yang sudah dipindahkan
app.config['UPLOAD_FOLDER'] = os.path.abspath(UPLOAD_FOLDER)
app.secret_key = 'super-secret-key-change-this'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
# ... (Blok INISIALISASI MODEL tetap SAMA PERSIS, tidak perlu diubah) ...
# ==============================================================================
# INISIALISASI MODEL (HANYA SEKALI SAAT APLIKASI START)
# ==============================================================================
print("="*50)
print("MEMUAT MODEL... Proses ini mungkin memakan waktu beberapa menit.")
print("="*50)
MODEL_NAME = "openai/whisper-large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
print(f"Whisper akan berjalan di: {DEVICE}")
transcriber = pipeline(
    task="automatic-speech-recognition", model=MODEL_NAME, device=DEVICE,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
print(">>> Model Whisper berhasil dimuat.")
try:
    with open('../data/the_quran_dataset.json', 'r', encoding='utf-8') as f: data = json.load(f)
    df = pd.DataFrame(data)
    df_latih = df[['surah_latin', 'ayah', 'latin', 'translation','arabic']].copy()
    df_latih.dropna(subset=['latin', 'translation'], inplace=True)
    df_latih = df_latih[df_latih['latin'].str.strip() != '']; df_latih = df_latih[df_latih['translation'].str.strip() != '']
    df_latih.reset_index(drop=True, inplace=True); corpus_latin_sempurna = df_latih['latin'].tolist()
    print(">>> Dataset Qur'an berhasil dimuat.")
except FileNotFoundError:
    print("KRITIS: File 'the_quran_dataset.json' tidak ditemukan di folder '../data/'."); exit()
model_path = '../model/model_quran_koreksi_gemparV3'
try:
    model_pencocokan = SentenceTransformer(model_path); print(">>> Model SentenceTransformer berhasil dimuat.")
except Exception as e:
    print(f"KRITIS: Gagal memuat model dari '{model_path}'. Pastikan path benar. Error: {e}"); exit()
print("Membuat indeks embeddings untuk pencarian..."); corpus_embeddings = model_pencocokan.encode(corpus_latin_sempurna, convert_to_tensor=True, show_progress_bar=True)
print(">>> Indeks embeddings siap.")
print("="*50); print("APLIKASI SIAP DIJALANKAN."); print("="*50)

# ... (Blok FUNGSI-FUNGSI LOGIKA tetap SAMA PERSIS, tidak perlu diubah) ...
ARABIC_TO_LATIN_QURAN = {
    # --- Huruf Dasar ---
    'ا': 'ā', 'أ': 'a', 'إ': 'i', 'آ': 'ā',  # Alif dan variannya
    'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j',
    'ح': 'ḥ',  # Ha' (lebih dalam)
    'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z',
    'س': 's', 'ش': 'sh',
    'ص': 'ṣ',  # Shad (emphatic s)
    'ض': 'ḍ',  # Dhad (emphatic d)
    'ط': 'ṭ',  # Ta' (emphatic t)
    'ظ': 'ẓ',  # Zha' (emphatic z)
    'ع': 'ʿ',  # 'Ayn (simbol khusus, lebih baik dari apostrof)
    'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
    'ه': 'h', 'و': 'w', 'ي': 'y',

    # --- Hamzah & Ta Marbuta ---
    'ء': 'ʾ',  # Hamzah (simbol khusus, lebih baik dari apostrof)
    'ئ': 'ʾ', 'ؤ': 'ʾ',
    'ة': 'h',  # Ta Marbuta

    # --- Harakat (Tanda Baca Vokal) ---
    'َ': 'a',  # Fathah
    'ُ': 'u',  # Dhammah
    'ِ': 'i',  # Kasrah
    'ً': 'an', # Fathatayn
    'ٌ': 'un', # Dhammatayn
    'ٍ': 'in', # Kasratayn

    # --- Tanda Sukun dan Tashdid (di-handle oleh logika, bukan kamus) ---
    'ْ': '',
    'ّ': '',

    # --- Vokal Panjang & Simbol Khusus Qur'an ---
    'ٰ': 'ā',  # Alif Khanjariyah (Alif kecil di atas)
    'ى': 'ā',  # Alif Maqsurah
    'ٓ': 'ā',  # Maddah
    'ٱ': '',   # Alif Waslah

    # --- Kata Khusus ---
    'اللّٰه': 'Allāh',

    # --- Tanda Baca & Angka ---
    '،': ',', '؛': ';', '؟': '?', '۔': '.',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

# Definisi huruf Shamsiyyah (Matahari)
SUN_LETTERS = "تثدذرزسشصضطظلن"

def transliterate_quranic(text):
    """
    Transliterasi teks Arab Qur'an ke Latin dengan menangani aturan-aturan khusus
    seperti Tashdid (Shaddah) dan Alif Lam (Shamsiyyah & Qamariyyah).
    """
    result = ""
    # Normalisasi beberapa karakter untuk konsistensi
    text = text.replace('ﷲ', 'اللّٰه')
    
    # Iterasi melalui setiap karakter dalam teks dengan indeksnya
    for i, char in enumerate(text):
        
        # Aturan 1: Tashdid / Shaddah (Menggandakan Konsonan)
        if char == 'ّ':
            # Cari mundur untuk menemukan konsonan terakhir yang ditambahkan
            for j in range(len(result) - 1, -1, -1):
                if result[j].isalpha() and result[j] not in "aiuāʾʿ": # Pastikan itu konsonan
                    # Gandakan konsonan tersebut
                    result += result[j]
                    break
            continue # Lanjutkan ke karakter berikutnya, jangan proses Shaddah lebih lanjut

        # Aturan 2: Alif Lam Shamsiyyah (Matahari)
        if char == 'ل' and i > 0 and text[i-1] in 'اٱ':
            # Lihat karakter setelah Lam (mungkin ada shaddah di antaranya)
            next_char_index = i + 1
            if next_char_index < len(text) and text[next_char_index] == 'ّ':
                next_char_index += 1
            
            if next_char_index < len(text) and text[next_char_index] in SUN_LETTERS:
                # Jika setelahnya adalah Huruf Matahari, 'l' tidak dibaca (lebur).
                # Jadi, jangan tambahkan 'l' ke hasil.
                pass 
            else:
                # Jika Huruf Bulan (Qamariyyah), 'l' dibaca jelas.
                result += 'l'
            continue # Lanjutkan ke karakter berikutnya

        # Aturan 3: Pemetaan Karakter Standar dari kamus
        result += ARABIC_TO_LATIN_QURAN.get(char, char)
        
    return result

def transcribe_audio(file_path):
    """
    Proses file audio, transkripsi dengan Whisper, dan transliterasi
    dengan aturan Qur'an yang disempurnakan.
    """
    try:
        # Pydub bisa menangani konversi dari webm jika ffmpeg terinstall
        if not file_path.lower().endswith('.wav'):
             audio_segment = AudioSegment.from_file(file_path)
             audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
             # Ganti nama file untuk disimpan sebagai WAV
             temp_wav_path = os.path.splitext(file_path)[0] + ".wav"
             audio_segment.export(temp_wav_path, format="wav")
             file_path = temp_wav_path

        audio, sr = librosa.load(file_path, sr=None, mono=True)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            
        result = transcriber(audio, generate_kwargs={"language": "arabic"})
        arabic_text = result["text"]
        
        # Menggunakan fungsi transliterasi baru yang lebih akurat
        latin_text = transliterate_quranic(arabic_text)
        
        return {
            "arabic": arabic_text.strip(),
            "latin": latin_text.strip(),
            "duration": f"{len(audio)/SAMPLE_RATE:.2f} detik"
        }
        
    except Exception as e:
        print(f"Error dalam transkripsi: {e}")
        return {"error": str(e)}

def cari_hybrid(query_tidak_sempurna, top_k=3, n_candidates=50):
    start_time = time.time()
    if not query_tidak_sempurna: return {"error": "Query tidak boleh kosong."}
    jaro_scores = [(idx, jellyfish.jaro_winkler_similarity(query_tidak_sempurna.lower(), ayat.lower())) for idx, ayat in enumerate(corpus_latin_sempurna)]
    jaro_scores.sort(key=lambda x: x[1], reverse=True); top_candidate_indices = [idx for idx, score in jaro_scores[:n_candidates]]
    candidate_embeddings = corpus_embeddings[top_candidate_indices]
    query_embedding = model_pencocokan.encode(query_tidak_sempurna, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
    final_results = []; 
    for i, score in enumerate(cosine_scores): final_results.append({'index': top_candidate_indices[i], 'score': score.item()})
    final_results.sort(key=lambda x: x['score'], reverse=True)
    top_matches = []
    for i in range(min(top_k, len(final_results))):
        result = final_results[i]; idx = result['index']; score = result['score']
        ayat_info = df_latih.loc[idx]
        top_matches.append({
            'latin': str(ayat_info['latin']),
            'arabic': str(ayat_info['arabic']),
            'translation': str(ayat_info['translation']),
            'surah': str(ayat_info['surah_latin']),
            'ayah': int(ayat_info['ayah']),   # <-- UBAH menjadi int() standar
            'score': f"{score:.4f}"
        })
    total_time = time.time() - start_time; print(f"Pencarian selesai dalam {total_time:.2f} detik.")
    return top_matches
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================================================================
# ROUTING APLIKASI FLASK (Rute lama tidak berubah)
# ==============================================================================
@app.route('/')
def index():
    """
    Hanya bertanggung jawab untuk menampilkan halaman HTML utama.
    Tidak ada logika POST di sini.
    """
    return render_template('index.html')

@app.route('/search_text', methods=['POST'])
def search_text_api():
    """
    Endpoint API KHUSUS untuk menerima query teks dari JavaScript.
    Mengembalikan hasil dalam format JSON.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Request tidak valid. Harap sertakan 'query' dalam body JSON."}), 400

    latin_query = data.get('query', '').strip()
    if not latin_query:
        return jsonify({"error": "Kueri pencarian tidak boleh kosong."}), 400

    # Panggil fungsi logika pencarian Anda
    search_results = cari_hybrid(latin_query)
    
    return jsonify({
        'query': latin_query,
        'search': search_results
    })

@app.route('/process_realtime_audio', methods=['POST'])
def process_realtime_audio_api():
    """
    Endpoint API KHUSUS untuk memproses rekaman audio dari browser.
    """
    if 'audio_data' not in request.files:
        return jsonify({"error": "Tidak ada data audio ditemukan"}), 400

    audio_file = request.files['audio_data']
    
    # Buat nama file yang unik untuk menghindari konflik
    filename = str(uuid.uuid4()) + ".webm"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        audio_file.save(filepath)
        # Panggil fungsi logika transkripsi Anda
        transcription_result = transcribe_audio(filepath)
    finally:
        # Pastikan file sementara selalu dihapus
        if os.path.exists(filepath):
            os.remove(filepath)
        
    if "error" in transcription_result:
        return jsonify({"error": transcription_result['error']}), 500
        
    latin_query = transcription_result.get('latin', '').strip()
    search_results = []
    if latin_query:
        # Panggil fungsi logika pencarian Anda
        search_results = cari_hybrid(latin_query)
    else:
        transcription_result['latin'] = "(Audio tidak dapat diinterpretasikan)"

    return jsonify({
        'transcription': transcription_result,
        'search': search_results
    })
# ==============================================================================
# ENDPOINT BARU UNTUK AUDIO REAL-TIME
# ==============================================================================
@app.route('/process_realtime_audio', methods=['POST'])
def process_realtime_audio():
    if 'audio_data' not in request.files:
        return jsonify({"error": "Tidak ada data audio ditemukan"}), 400

    audio_file = request.files['audio_data']
    
    # Buat nama file yang unik untuk menghindari konflik
    filename = str(uuid.uuid4()) + ".webm"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    # Gunakan kembali fungsi yang sudah ada
    transcription_result = transcribe_audio(filepath)
    
    if "error" in transcription_result:
        return jsonify({"error": transcription_result['error']}), 500
        
    latin_query = transcription_result.get('latin', '').strip()
    search_results = []
    if latin_query:
        search_results = cari_hybrid(latin_query)

    # Siapkan data untuk dikirim kembali sebagai JSON
    response_data = {
        'transcription': transcription_result,
        'search': search_results,
        'query': latin_query
    }
    
    return jsonify(response_data)


if __name__ == '__main__':
    # Pastikan server berjalan dengan SSL context jika dideploy,
    # karena getUserMedia butuh koneksi aman (https) atau localhost.
    # Untuk pengembangan lokal, http://127.0.0.1 sudah cukup.
    app.run(debug=True, use_reloader=False)