from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, db
import os

# Inisialisasi Flask
app = Flask(__name__)

# 🔥 Load credentials Firebase (ganti dengan path file sertifikat Firebase-mu)
cred = credentials.Certificate("ba-nanasapplication-firebase-adminsdk-fbsvc-ac57e198a8.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ba-nanasapplication-default-rtdb.firebaseio.com/'
})


MODEL_URL = "https://www.dropbox.com/scl/fi/zare7lltsj94cwjlmb6z9/PISANG16CLASS.h5?rlkey=5ohf9ddgxo1j8753tzx8igtim&st=9pi4643l&dl=1"

# 🔥 Load model (ganti MODEL_PATH jika diperlukan)
MODEL_PATH = "models/PISANG16CLASS.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")
model = load_model(MODEL_PATH)

# 🔥 Daftar label kelas
class_labels = [
    "ambon_hampirbusuk", "ambon_matang", "ambon_mentah", "ambon_setengahmatang", "ambon_terlalumatang",
    "kepok_hampirbusuk", "kepok_matang", "kepok_mentah", "kepok_setengahmatang", "kepok_terlalumatang",
    "non_banana",
    "susu_hampirbusuk", "susu_matang", "susu_mentah", "susu_setengahmatang", "susu_terlalumatang",
]

# 🔥 Mapping kategori ke rentang nilai kematangan
category_ranges = {
    "mentah": (0, 20),
    "setengahmatang": (20, 40),
    "matang": (40, 60),
    "terlalumatang": (60, 80),
    "hampirbusuk": (80, 100)
}

# 🔥 Mapping hari dan pesan kematangan
ripening_info = {
    "mentah": {"days": 5, "message": "Pisang masih mentah, perlu sekitar 5 hari untuk matang."},
    "setengahmatang": {"days": 3, "message": "Pisang setengah matang, akan matang dalam 3 hari."},
    "matang": {"days": 0, "message": "Pisang sudah matang, siap dikonsumsi."},
    "terlalumatang": {"days": -1, "message": "Pisang terlalu matang, sebaiknya segera dikonsumsi."},
    "hampirbusuk": {"days": -3, "message": "Pisang hampir busuk, tidak layak dikonsumsi."},
}

# 🔥 Mapping range glukosa dan kalori berdasarkan jenis pisang dan tingkat kematangannya
glukosa_kalori_data = {
    ("susu", "mentah"): {"glukosa": "0.75 - 4.00", "kalori": "4.8 - 25.6"},
    ("susu", "setengahmatang"): {"glukosa": "3.75 - 4.00", "kalori": "24.0 - 25.6"},
    ("susu", "matang"): {"glukosa": "8.25 - 10.5", "kalori": "52.8 - 67.2"},
    ("susu", "terlalumatang"): {"glukosa": "9.00 - 10.5", "kalori": "57.6 - 67.2"},
    ("susu", "hampirbusuk"): {"glukosa": "11.5", "kalori": "73.6"},
    ("kepok", "mentah"): {"glukosa": "1.25 - 4.00", "kalori": "8.0 - 25.6"},
    ("kepok", "setengahmatang"): {"glukosa": "3.25 - 5.75", "kalori": "20.8 - 36.8"},
    ("kepok", "matang"): {"glukosa": "5.50 - 11.25", "kalori": "35.2 - 72.0"},
    ("kepok", "terlalumatang"): {"glukosa": "9.50 - 11.50", "kalori": "60.8 - 73.6"},
    ("kepok", "hampirbusuk"): {"glukosa": " > 11.50", "kalori": " > 73.6"},
    ("ambon", "mentah"): {"glukosa": "0.75 - 1.25", "kalori": "4.8 - 8.0"},
    ("ambon", "setengahmatang"): {"glukosa": "6.00 - 7.00", "kalori": "38.4 - 44.8"},
    ("ambon", "matang"): {"glukosa": "7.00 - 11.25", "kalori": "44.8 - 72.0"},
    ("ambon", "terlalumatang"): {"glukosa": "7.50 - 11.25", "kalori": "48.0 - 72.0"},
    ("ambon", "hampirbusuk"): {"glukosa": " > 11.25", "kalori": " > 72.0"},
}

# Fungsi untuk memisahkan jenis dan tingkat kematangan
def separate_type_and_ripeness(class_label):
    label_parts = class_label.split('_')
    if len(label_parts) == 2:
        return label_parts[0], label_parts[1]
    else:
        raise ValueError(f"Label '{class_label}' tidak dalam format yang diharapkan.")

# Fungsi untuk menghitung nilai kematangan berdasarkan prediksi
def calculate_ripeness_value(prediction, categories):
    prediction = tf.nn.softmax(prediction).numpy()
    class_idx = np.argmax(prediction)
    predicted_category = categories[class_idx]

    if predicted_category == "non_banana":
        return {"error": "Gambar bukan pisang."}
    if predicted_category not in categories:
        return None

    predicted_type, predicted_ripeness = separate_type_and_ripeness(predicted_category)
    lower_bound, upper_bound = category_ranges.get(predicted_ripeness, (0, 100))
    confidence = float(prediction[class_idx])
    ripeness_value = lower_bound + confidence * (upper_bound - lower_bound)
    ripening_details = ripening_info.get(predicted_ripeness, {"days": "N/A", "message": "Tidak diketahui"})
    glukosa_kalori = glukosa_kalori_data.get((predicted_type, predicted_ripeness), {"glukosa": "N/A", "kalori": "N/A"})

    return {
        "jenis_pisang": predicted_type,
        "kategori": predicted_ripeness,
        "nilai_kematangan": round(ripeness_value, 2),
        "probabilitas": round(confidence * 100, 2),
        "hari_untuk_matang": ripening_details["days"],
        "pesan_untuk_matang": ripening_details["message"],
        "range_glukosa": glukosa_kalori["glukosa"],
        "range_kalori": glukosa_kalori["kalori"]
    }

# Endpoint POST /predict untuk memproses gambar dan menyimpan hasil ke Firebase
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "File gambar harus disertakan"}), 400
    file = request.files["file"]
    try:
        img = Image.open(BytesIO(file.read())).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        prediction = model.predict(img_array)[0]
        ripeness_result = calculate_ripeness_value(prediction, class_labels)
        if ripeness_result is None:
            return jsonify({"error": "Gambar bukan pisang atau tidak dikenali"}), 400

        # Jika ada error (misalnya, gambar bukan pisang), jangan simpan
        if "error" in ripeness_result:
            return jsonify(ripeness_result), 400

        # Simpan ke Firebase di node /predictions
        ref = db.reference("/predictions")
        new_ref = ref.push(ripeness_result)
        # Tambahkan key sebagai id pada data
        new_ref.update({"id": new_ref.key})
        return jsonify(ripeness_result)
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 400

# Endpoint GET /history untuk mengambil semua data history
@app.route("/history", methods=["GET"])
def get_history():
    try:
        ref = db.reference("/predictions")
        data = ref.get()
        if data is None:
            data = {}
        history_list = []
        for key, val in data.items():
            if "id" not in val:
                val["id"] = key
            history_list.append(val)
        return jsonify(history_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint DELETE /history/<item_id> untuk menghapus data history tertentu
@app.route("/history/<string:item_id>", methods=["DELETE"])
def delete_history_item(item_id):
    try:
        ref = db.reference("/predictions").child(item_id)
        ref.delete()
        return jsonify({"message": "Deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
