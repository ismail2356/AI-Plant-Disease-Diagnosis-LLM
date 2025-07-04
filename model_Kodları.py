"""
Bitki Hastalığı Sınıflandırma ve Rehberlik Sistemi
- Google ViT (Vision Transformer) modeli kullanarak bitki hastalıklarını tespit etme
- LLM (Google Gemini API) entegrasyonu ile bitki hastalığı rehberliği
"""

# Gerekli kütüphanelerin yüklenmesi
!pip install -q transformers datasets timm scikit-learn pandas matplotlib seaborn google-generativeai pillow

# Kütüphanelerin import edilmesi
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from transformers import ViTForImageClassification, ViTConfig, ViTFeatureExtractor, get_linear_schedule_with_warmup
import google.generativeai as genai
import time
import random
import json
from tqdm.auto import tqdm

# Kullanılan cihazın (CPU/GPU) belirlenmesi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Sabit değişkenlerin tanımlanması
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WARMUP_STEPS = 500
MODEL_NAME = "google/vit-base-patch16-224-in21k"  # Projede kullanılan ViT modeli
KAGGLE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"  # Veri seti dizini

# Yeniden üretilebilirlik için seed ayarı
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(SEED)

# Veri dosyalarının hazırlanması
def setup_data():
    print("Veri dosyaları hazırlanıyor...")
    
    # Dosya yollarının kontrol edilmesi
    train_csv_path = os.path.join(KAGGLE_DIR, "train.csv")
    test_csv_path = os.path.join(KAGGLE_DIR, "test.csv")
    train_images_path = os.path.join(KAGGLE_DIR, "images")
    
    # Veri setinin okunması
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Hastalık kategorilerinin belirlenmesi
    categories = train_df.columns[1:].tolist()
    
    # İstatistikleri gösterme
    print(f"Eğitim verisi boyutu: {train_df.shape}")
    print(f"Test verisi boyutu: {test_df.shape}")
    print(f"Hastalık kategorileri: {categories}")
    
    # Kategori dağılımının görselleştirilmesi
    plt.figure(figsize=(10, 6))
    train_df[categories].sum().sort_values().plot(kind='barh')
    plt.title('Eğitim Veri Setindeki Hastalık Dağılımı')
    plt.tight_layout()
    plt.savefig('hastalık_dağılımı.png')
    plt.close()
    
    return train_df, test_df, train_images_path, categories

# Veri setinin tanımlanması
class PlantDataset(Dataset):
    def __init__(self, dataframe, img_dir, categories, feature_extractor=None, mode='train', transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.categories = categories
        self.feature_extractor = feature_extractor
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.feature_extractor:
            image = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        if self.mode == 'test':
            return image
        else:
            labels = torch.tensor(self.dataframe.iloc[idx][self.categories].values.astype(np.float32))
            return image, labels

# Veri artırma ve dönüşüm işlemleri
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ViT modeli oluşturma
class PlantDiseaseModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(PlantDiseaseModel, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

# Eğitim döngüsü
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Eğitim")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

# Değerlendirme döngüsü
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Değerlendirme"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    roc_auc = roc_auc_score(all_labels, all_preds, average='macro')
    
    return total_loss / len(dataloader), roc_auc, all_preds, all_labels

# Tahmin işlemi
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Tahmin"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
    
    return np.vstack(all_preds)

# Sonuçların görselleştirilmesi
def visualize_results(val_labels, val_preds, categories):
    # Karışıklık matrisi için eşik değeri belirle
    threshold = 0.5
    val_preds_binary = (val_preds > threshold).astype(int)
    
    # Her sınıf için ROC eğrisi
    plt.figure(figsize=(15, 10))
    for i, category in enumerate(categories):
        fpr, tpr, _ = roc_curve(val_labels[:, i], val_preds[:, i])
        plt.plot(fpr, tpr, label=f'{category} (AUC = {roc_auc_score(val_labels[:, i], val_preds[:, i]):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Sınıflandırma raporu
    print("Sınıflandırma Raporu:")
    for i, category in enumerate(categories):
        print(f"\n{category}:")
        print(classification_report(val_labels[:, i], val_preds_binary[:, i]))
    
    # Örnek görsellerin tahminlerini görüntüleme
    return val_preds_binary

# LLM entegrasyonu için rehberlik sistemi
def setup_llm_guidance():
    try:
        # Google Gemini API anahtarı
        # Gerçek uygulamada bu anahtarı güvenli bir şekilde saklamalısınız
        # API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY" 
        # genai.configure(api_key=API_KEY)
        
        print("LLM rehberlik sistemi kurulumu başarılı.")
        print("Not: Gerçek API anahtarınızı eklemeniz gerekecek.")
        
        # LLM model için prompt şablonu örneği
        llm_prompt_template = """
        Aşağıdaki bitki hastalığı tespit edilmiştir:
        
        Bitki: {plant_type}
        Tespit Edilen Hastalık: {disease_name}
        Güven Skoru: {confidence:.2f}
        
        Lütfen bu hastalık hakkında aşağıdaki bilgileri sağlayın:
        1. Hastalık belirtileri
        2. Hastalığın yaygın nedenleri
        3. Bu hastalığı tedavi etmek için önerilen yöntemler
        4. Gelecekte bu hastalığı önlemek için alınması gereken tedbirler
        """
        
        # Örnek hastalık bilgi tabanı
        disease_info = {
            "healthy": "Sağlıklı bitki yaprakları",
            "apple_scab": "Elma karalekesi, elmada yaygın görülen bir mantar hastalığıdır",
            "cedar_apple_rust": "Sedir elma pası, elmada görülen mantar hastalığıdır",
            "complex": "Çoklu hastalık belirtileri görülmektedir"
        }
        
        # Örnek rehberlik yanıtı oluşturma
        sample_response = llm_prompt_template.format(
            plant_type="Elma",
            disease_name="Elma Karalekesi (Apple Scab)",
            confidence=0.92
        )
        
        print("\nLLM Rehberlik Örneği:")
        print(sample_response)
        
        return disease_info
        
    except Exception as e:
        print(f"LLM entegrasyonu hazırlanırken hata oluştu: {e}")
        return None

# Ana fonksiyon
def main():
    print("Bitki Hastalığı Sınıflandırma ve Rehberlik Sistemi başlatılıyor...")
    
    # Veri setinin hazırlanması
    train_df, test_df, train_images_path, categories = setup_data()
    
    # Eğitim ve validasyon veri setlerinin ayrılması
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df[categories].values)
    
    print(f"Eğitim veri seti boyutu: {train_data.shape}")
    print(f"Validasyon veri seti boyutu: {val_data.shape}")
    
    # Feature extractor ve dönüşümlerin hazırlanması
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    train_transform, val_transform = get_transforms()
    
    # Veri setlerinin oluşturulması
    train_dataset = PlantDataset(
        dataframe=train_data, 
        img_dir=os.path.join(train_images_path), 
        categories=categories,
        transform=train_transform
    )
    
    val_dataset = PlantDataset(
        dataframe=val_data, 
        img_dir=os.path.join(train_images_path), 
        categories=categories,
        transform=val_transform
    )
    
    test_dataset = PlantDataset(
        dataframe=test_df, 
        img_dir=os.path.join(train_images_path), 
        categories=categories,
        mode='test',
        transform=val_transform
    )
    
    # Veri yükleyicilerin oluşturulması
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Modelin oluşturulması
    model = PlantDiseaseModel(MODEL_NAME, len(categories))
    model.to(device)
    
    # Kayıp fonksiyonu, optimizer ve scheduler tanımlanması
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Eğitim döngüsü
    best_auc = 0
    train_losses = []
    val_losses = []
    val_aucs = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Eğitim
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        train_losses.append(train_loss)
        
        # Değerlendirme
        val_loss, val_auc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        print(f"Eğitim Kaybı: {train_loss:.4f}, Validasyon Kaybı: {val_loss:.4f}, Validasyon ROC AUC: {val_auc:.4f}")
        
        # En iyi modelin kaydedilmesi
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"En iyi model kaydedildi! Validasyon ROC AUC: {val_auc:.4f}")
    
    # Eğitim performansının görselleştirilmesi
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(val_losses, label='Validasyon Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Validasyon ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.close()
    
    # En iyi model ile validasyon sonuçlarının görselleştirilmesi
    model.load_state_dict(torch.load('best_model.pth'))
    _, final_auc, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
    print(f"\nEn iyi model - Validasyon ROC AUC: {final_auc:.4f}")
    
    val_preds_binary = visualize_results(final_labels, final_preds, categories)
    
    # Test veri seti üzerinde tahmin
    test_preds = predict(model, test_loader, device)
    
    # Tahminlerin kaydedilmesi
    submission_df = pd.DataFrame(data=test_preds, columns=categories)
    submission_df['image_id'] = test_df['image_id'].values
    submission_df = submission_df[['image_id'] + categories]
    submission_df.to_csv('submission.csv', index=False)
    print("\nTahminler 'submission.csv' dosyasına kaydedildi.")
    
    # LLM rehberlik sisteminin kurulması
    disease_info = setup_llm_guidance()
    
    # Örnek rehberlik çıktısı
    if disease_info:
        print("\nÖrnek Tanı ve Rehberlik:")
        # Örnek bir tahmin için rehberlik sağlama
        sample_idx = 0
        sample_img_id = val_data.iloc[sample_idx]['image_id']
        pred_disease_idx = np.argmax(final_preds[sample_idx])
        pred_disease = categories[pred_disease_idx]
        confidence = final_preds[sample_idx][pred_disease_idx]
        
        print(f"Görüntü ID: {sample_img_id}")
        print(f"Tahmin Edilen Hastalık: {pred_disease}")
        print(f"Güven Skoru: {confidence:.4f}")
        print(f"Hastalık Bilgisi: {disease_info.get(pred_disease, 'Bilgi bulunamadı')}")
        
        # Gerçek bir uygulamada burada LLM API çağrısı yapılacaktır
        print("\nNot: Gerçek bir uygulamada, LLM API çağrısı yapılarak daha detaylı rehberlik sağlanacaktır.")
    
    print("\nBitki Hastalığı Sınıflandırma ve Rehberlik Sistemi tamamlandı.")

if __name__ == "__main__":
    main() 