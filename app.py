import os
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import uuid
import datetime
from transformers import ViTForImageClassification, ViTImageProcessor
import json
from gemini_helper import GeminiHelper  # Gemini yardımcı sınıfını içe aktar

app = Flask(__name__)

# Model ve işlemci yolları
MODEL_PATH = "model"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')

# Upload klasörünün varlığını kontrol et
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Türkçe bitki ve hastalık isimleri sözlüğü
turkish_names = {
    # Bitkiler
    "Apple": "Elma",
    "Grape": "Üzüm",
    "Pepper,_bell": "Biber",
    "Tomato": "Domates",
    
    # Hastalıklar
    "healthy": "Sağlıklı",
    "Apple_scab": "Elma Uyuzu",
    "Black_rot": "Siyah Çürüklük",
    "Cedar_apple_rust": "Sedir Elma Pası",
    "Esca_(Black_Measles)": "Esca (Siyah Kızamık)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Yaprak Yanıklığı",
    "Bacterial_spot": "Bakteriyel Leke",
    "Early_blight": "Erken Yanıklık",
    "Late_blight": "Geç Yanıklık",
    "Leaf_Mold": "Yaprak Küfü",
    "Septoria_leaf_spot": "Septoria Yaprak Lekesi",
    "Spider_mites Two-spotted_spider_mite": "Örümcek Akarı",
    "Target_Spot": "Hedef Nokta",
    "Tomato_Yellow_Leaf_Curl_Virus": "Sarı Yaprak Kıvırcıklığı Virüsü",
    "Tomato_mosaic_virus": "Mozaik Virüsü"
}

# Model ve işlemciyi yükle
def load_model():
    try:
        # Model ve processor'ı yükle
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        
        # Label bilgilerini yükle
        with open(os.path.join(MODEL_PATH, 'label_info.json'), 'r') as f:
            label_info = json.load(f)
        
        # GPU varsa kullan
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        return model, processor, label_info
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        return None, None, None

model, processor, label_info = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image):
    try:
        # Görüntüyü işle
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Sonuçları işle
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.topk(probs, k=3)
        
        results = []
        for prob, idx in zip(predictions.values[0], predictions.indices[0]):
            class_name = label_info['classes'][idx]
            plant, condition = class_name.split('___')
            
            # Türkçe bitki ve hastalık isimlerini bul
            turkish_plant = turkish_names.get(plant, plant)
            turkish_condition = turkish_names.get(condition, condition)
            
            results.append({
                'disease': class_name,
                'probability': float(prob) * 100,
                'plant': plant,
                'condition': condition,
                'turkish_plant': turkish_plant,
                'turkish_condition': turkish_condition
            })
        
        return results
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        return None

def get_disease_info(plant, condition):
    """
    Gemini API'yi kullanarak hastalık bilgisi al
    """
    try:
        # Sağlıklı durum için bilgi döndürme
        if condition.lower() == "healthy" or condition.lower() == "sağlıklı":
            return f"""BİTKİ SAĞLIĞI DURUMU

Bitkiniz sağlıklı görünüyor. Herhangi bir hastalık belirtisi tespit edilmedi.

BİTKİ BAKIMI İÇİN ÖNERİLER

* Düzenli sulama yapın
* Uygun gübreleme programı uygulayın
* Periyodik olarak yaprak kontrolü yapın
* Zararlı böceklere karşı önlem alın
"""
        
        # Hastalık durumu için Gemini'den bilgi al
        # use_cache=True parametresi ekleyerek önbellekleme aktif edildi
        disease_info = GeminiHelper.get_disease_info(
            disease_name=condition,
            plant_type=plant,
            use_cache=True,
            cache_max_age=86400  # 24 saat boyunca önbellekte tut
        )
        
        # Yanıtı formatla ve düzenle
        formatted_info = disease_info.strip()
        
        # Eğer yanıtta hata mesajı varsa
        if "hata oluştu" in formatted_info.lower():
            return "Hastalık hakkında detaylı bilgi alınamadı. Lütfen bir ziraat mühendisine danışın."
            
        return formatted_info
    except Exception as e:
        print(f"Hastalık bilgisi alma hatası: {str(e)}")
        return "Hastalık hakkında detaylı bilgi alınamadı. Lütfen bir ziraat mühendisine danışın."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Geçersiz dosya formatı'}), 400
    
    try:
        # Görüntüyü oku
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Dosyayı kaydetmek için benzersiz isim oluştur
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4().hex)[:8]
        extension = file.filename.rsplit('.', 1)[1].lower()
        image_filename = f"leaf_{timestamp}_{unique_id}.{extension}"
        
        # Görüntüyü kaydet
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image.save(image_path)
        
        # Tahmin yap
        results = predict_image(image)
        
        if results is None:
            return jsonify({'error': 'Tahmin yapılamadı'}), 500
        
        # Sadece en yüksek olasılıklı sonucu al
        top_result = results[0]
        
        # Hastalık hakkında detaylı bilgi al
        disease_info = get_disease_info(
            plant=top_result['turkish_plant'],
            condition=top_result['turkish_condition']
        )
        
        # Sonuç sayfasını render et
        return render_template('result.html', 
                               result=top_result, 
                               disease_info=disease_info,
                               image_filename=image_filename)
    
    except Exception as e:
        return jsonify({'error': f'Bir hata oluştu: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
