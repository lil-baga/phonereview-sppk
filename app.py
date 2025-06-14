import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from math import ceil

# ---------------------------------------------------------------------------
# BAGIAN 0: KONFIGURASI APLIKASI FLASK
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'supersecretkey_for_flash_messages'
app.config['UPLOAD_FOLDER'] = 'uploads'
CONFIG_FILE = 'config.json'
ITEMS_PER_PAGE = 10

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ---------------------------------------------------------------------------
# BAGIAN 1: IMPLEMENTASI LOGIKA FUZZY (Tidak ada perubahan)
# ---------------------------------------------------------------------------
class TFN:
    def __init__(self, l, m, u): self.l, self.m, self.u = l, m, u
    def __repr__(self): return f"({self.l:.2f}, {self.m:.2f}, {self.u:.2f})"
    def defuzzify(self): return (self.l + self.m + self.u) / 3
    def __add__(self, other): return TFN(self.l + other.l, self.m + other.m, self.u + other.u)
    def __mul__(self, scalar): return TFN(self.l * scalar, self.m * scalar, self.u * scalar)
    @staticmethod
    def divide(t1, t2):
        l2 = t2.l if t2.l != 0 else 1e-9
        m2 = t2.m if t2.m != 0 else 1e-9
        u2 = t2.u if t2.u != 0 else 1e-9
        return TFN(t1.l / u2, t1.m / m2, t1.u / l2)

LINGUISTIC_SCALE = {
    "Sangat Rendah": TFN(0, 0.1, 0.2), "Rendah": TFN(0.1, 0.3, 0.5),
    "Cukup": TFN(0.3, 0.5, 0.7), "Tinggi": TFN(0.5, 0.7, 0.9),
    "Sangat Tinggi": TFN(0.7, 0.9, 1.0)
}

def crisp_to_linguistic_term(value, min_val, max_val):
    if max_val == min_val: norm_val = 0.5
    else: norm_val = (value - min_val) / (max_val - min_val)
    if norm_val < 0.2: return "Sangat Rendah"
    elif norm_val < 0.4: return "Rendah"
    elif norm_val < 0.6: return "Cukup"
    elif norm_val < 0.8: return "Tinggi"
    else: return "Sangat Tinggi"

def get_fuzzy_matrix(df, criteria_config):
    fuzzy_data, linguistic_data = [], []
    for _, row in df.iterrows():
        fuzzy_row, linguistic_row = [], []
        for col_name, config in criteria_config.items():
            value = row[col_name]
            min_val, max_val = df[col_name].min(), df[col_name].max()
            if config['type'] == 'Cost':
                rev_value = max_val - (value - min_val)
                term = crisp_to_linguistic_term(rev_value, min_val, max_val)
            else:
                term = crisp_to_linguistic_term(value, min_val, max_val)
            linguistic_row.append(term)
            fuzzy_row.append(LINGUISTIC_SCALE[term])
        fuzzy_data.append(fuzzy_row)
        linguistic_data.append(linguistic_row)
    return fuzzy_data, linguistic_data

# ---------------------------------------------------------------------------
# BAGIAN 2: IMPLEMENTASI FUZZY SAW & ARAS (Tidak ada perubahan)
# ---------------------------------------------------------------------------
def fuzzy_saw(fuzzy_matrix, weights):
    weighted_matrix = [[tfn * w for tfn, w in zip(row, weights)] for row in fuzzy_matrix]
    # Koreksi: Penjumlahan TFN perlu inisialisasi TFN(0,0,0)
    final_scores = []
    for row in weighted_matrix:
        score = TFN(0,0,0)
        for tfn in row:
            score += tfn
        final_scores.append(score)
    return [s.defuzzify() for s in final_scores], final_scores

def fuzzy_aras(fuzzy_matrix, weights):
    optimal_row = [TFN(1.0, 1.0, 1.0)] * len(weights)
    extended_matrix = [optimal_row] + fuzzy_matrix
    optimality_values = []
    for row in extended_matrix:
        s_i = TFN(0,0,0)
        for tfn, w in zip(row, weights):
            s_i += (tfn * w)
        optimality_values.append(s_i)
    s_optimal, s_alternatives = optimality_values[0], optimality_values[1:]
    utility_degrees = [TFN.divide(s_alt, s_optimal) for s_alt in s_alternatives]
    return [k.defuzzify() for k in utility_degrees], utility_degrees

# ---------------------------------------------------------------------------
# BAGIAN 3: FUNGSI BANTU UNTUK KONFIGURASI (Tidak ada perubahan)
# ---------------------------------------------------------------------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"last_csv": None, "criteria": {}}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# ---------------------------------------------------------------------------
# BAGIAN 4: FLASK ROUTES (LOGIKA DIPERBARUI)
# ---------------------------------------------------------------------------
def paginate(df, page):
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    return df.iloc[start:end]

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    config = load_config()
    results = None
    pratinjau_page = request.args.get('pratinjau_page', 1, type=int)
    keputusan_page = request.args.get('keputusan_page', 1, type=int)
    linguistik_page = request.args.get('linguistik_page', 1, type=int)
    saw_page = request.args.get('saw_page', 1, type=int)
    aras_page = request.args.get('aras_page', 1, type=int)

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            config['last_csv'] = filename
            save_config(config)
            flash(f'File "{file.filename}" berhasil diunggah.', 'success')
        else:
            flash('Tidak ada file yang dipilih', 'warning')
        return redirect(url_for('dashboard'))

    if config.get('last_csv') and os.path.exists(config['last_csv']) and config.get('criteria'):
        try:
            df_raw = pd.read_csv(config['last_csv'])
            criteria_config = config['criteria']
            selected_criteria = list(criteria_config.keys())
            
            if not all(col in df_raw.columns for col in selected_criteria):
                flash(f"Error: Kriteria ({', '.join(selected_criteria)}) tidak ditemukan di file. Silakan perbarui pengaturan.", 'danger')
            else:
                decision_matrix_df = df_raw[selected_criteria]
                weights_list = np.array([v['weight'] for v in criteria_config.values()])
                
                if not np.isclose(weights_list.sum(), 1.0) and weights_list.sum() > 0:
                    weights_list = weights_list / weights_list.sum()

                fuzzy_matrix, linguistic_matrix = get_fuzzy_matrix(decision_matrix_df, criteria_config)
                linguistic_df = pd.DataFrame(linguistic_matrix, columns=selected_criteria, index=df_raw.index)

                saw_scores, _ = fuzzy_saw(fuzzy_matrix, weights_list)
                df_saw = pd.DataFrame({
                    'Nama HP': df_raw['model'],  # Ganti "Nama" jika nama kolom berbeda
                    'Skor Defuzzifikasi': saw_scores
                }).sort_values('Skor Defuzzifikasi', ascending=False)
                df_saw['Peringkat'] = range(1, len(df_saw) + 1)

                aras_scores, _ = fuzzy_aras(fuzzy_matrix, weights_list)
                df_aras = pd.DataFrame({
                    'Nama HP': df_raw['model'],  # Sesuaikan juga nama kolomnya
                    'Derajat Utilitas Defuzzifikasi': aras_scores
                }).sort_values('Derajat Utilitas Defuzzifikasi', ascending=False)
                df_aras['Peringkat'] = range(1, len(df_aras) + 1)
                
                results = {
                    "pratinjau": {
                        "data": paginate(df_raw, pratinjau_page),
                        "page": pratinjau_page,
                        "total_pages": ceil(len(df_raw) / ITEMS_PER_PAGE)
                    },
                    "keputusan": {
                        "data": paginate(decision_matrix_df.reset_index(), keputusan_page),
                        "page": keputusan_page,
                        "total_pages": ceil(len(decision_matrix_df) / ITEMS_PER_PAGE)
                    },
                    "linguistik": {
                        "data": paginate(linguistic_df.reset_index(), linguistik_page),
                        "page": linguistik_page,
                        "total_pages": ceil(len(linguistic_df) / ITEMS_PER_PAGE)
                    },
                    "saw": {
                        "data": paginate(df_saw, saw_page),
                        "page": saw_page,
                        "total_pages": ceil(len(df_saw) / ITEMS_PER_PAGE)
                    },
                    "aras": {
                        "data": paginate(df_aras, aras_page),
                        "page": aras_page,
                        "total_pages": ceil(len(df_aras) / ITEMS_PER_PAGE)
                    },
                }


        except Exception as e:
            flash(f"Terjadi error saat memproses file: {e}", "danger")

    return render_template('dashboard.html', config=config, results=results, current_page='dashboard')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    config = load_config()
    available_columns = []
    
    if config.get('last_csv') and os.path.exists(config['last_csv']):
        try:
            df = pd.read_csv(config['last_csv'])
            available_columns = df.select_dtypes(include=np.number).columns.tolist()
        except Exception as e:
            flash(f"Tidak dapat membaca file CSV terakhir: {e}", 'warning')

    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'select_criteria':
            selected_cols = request.form.getlist('selected_criteria')
            # Reset kriteria, simpan hanya yang dipilih dengan nilai default
            config['criteria'] = {col: {"type": "Benefit", "weight": 0} for col in selected_cols}
            save_config(config)
            flash('Kriteria berhasil dipilih. Sekarang tentukan tipe dan bobotnya.', 'info')
            return redirect(url_for('settings'))

        elif action == 'save_settings':
            new_criteria_settings = {}
            total_weight = 0
            # Ambil kriteria yang sudah ada di config untuk diproses
            current_criteria = config.get('criteria', {})
            for col_name in current_criteria.keys():
                weight_str = request.form.get(f'weight_{col_name}', '0')
                try:
                    weight = float(weight_str)
                    total_weight += weight
                    new_criteria_settings[col_name] = {
                        'type': request.form.get(f'type_{col_name}', 'Benefit'),
                        'weight': weight
                    }
                except ValueError:
                    flash(f"Nilai bobot untuk '{col_name}' tidak valid. Lewati.", 'warning')
            
            if not np.isclose(total_weight, 1.0) and total_weight > 0:
                flash(f"Peringatan: Total bobot adalah {total_weight:.2f}, bukan 1.0. Bobot akan dinormalisasi saat perhitungan.", 'warning')
            
            config['criteria'] = new_criteria_settings
            save_config(config)
            flash('Pengaturan tipe dan bobot berhasil disimpan!', 'success')
            return redirect(url_for('dashboard'))

    return render_template('settings.html', config=config, available_columns=available_columns, current_page='settings')

if __name__ == '__main__':
    app.run(debug=True)
