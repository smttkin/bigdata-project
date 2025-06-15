import pandas as pd
import numpy as np
df = pd.read_csv("global_terrorism_dataset_filtered.csv")

# Tüm ağırlıklar 1-10 aralığındadır
attack_type_1_weights = {
    1: 7,   # SUİKAST
    2: 6,   # SİLAHLI SALDIRI
    3: 9,   # BOMBA/PATLAYICI
    4: 8,   # UÇAK/ARAÇ KAÇIRMA
    5: 8,   # REHİNE ALMA (BARİKAT OLAYI)
    6: 7,   # REHİNE ALMA (KAÇIRMA)
    7: 8,   # TESİS/ALTYAPI SALDIRISI
    8: 4,   # SİLAHSIZ SALDIRI
    9: 5    # BİLİNMİYOR
}

weapon_weights = {
    1: 10,  # Biyolojik
    2: 9,   # Kimyasal
    3: 8,   # Radyolojik
    4: 10,  # Nükleer
    5: 7,   # Ateşli Silahlar
    6: 8,   # Patlayıcılar
    7: 2,   # Sahte Silahlar
    8: 7,   # Yanıcı Maddeler
    9: 4,   # Yakın Dövüş Silahları
    10: 6,  # Araç
    11: 5,  # Sabotaj Ekipmanı
    12: 5,  # Diğer
    13: 5   # Bilinmiyor
}

targtype1_weights = {
    1: 7,   # İŞLETMELER – Sıklıkla hedef, ekonomik/simgesel etkisi büyük
    2: 9,   # DEVLET (GENEL) – Değerli hedef, otorite simgesi
    3: 8,   # POLİS – İç güvenliğe doğrudan tehdit
    4: 9,   # ORDU – Stratejik güvenlik hedefi
    5: 6,   # KÜRTAJLA İLGİLİ – Niş ama ideolojik hassas
    6: 8,   # HAVALİMANI/UÇAK – Yüksek kayıp potansiyeli, uluslararası yankı
    7: 9,   # DEVLET (DİPLOMATİK) – Yüksek simgesel ve dış politika etkisi
    8: 7,   # EĞİTİM KURUMLARI – Sivil hedef, ideolojik hassas
    9: 5,   # GIDA/SU TEDARİĞİ – Stratejik ama nadir hedef
    10: 7,  # MEDYA – İfade özgürlüğü sembolü, orta düzey sıklık
    11: 6,  # DENİZCİLİK – Orta etkili, bölgesel yoğunlukta
    12: 6,  # STK’LAR – İnsani yardım, politik anlam
    13: 5,  # DİĞER – Belirsiz/az sıklıkta hedef
    14: 10, # SİVİLLER/MÜLKLER – En çok hedef alınan grup
    15: 7,  # DİNİ KURUM/KİŞİLER – İdeolojik motivasyon güçlü
    16: 5,  # TELEKOMÜNİKASYON – Altyapı hedefi, orta sıklıkta
    17: 6,  # TERÖRİSTLER/GAYRİ RESMİ GRUPLAR – Nadir ama doğrudan çatışma
    18: 6,  # TURİSTLER – Savunmasız hedefler, medya ilgisi yüksek
    19: 8,  # ULAŞIM (HAVACILIK HARİÇ) – Yüksek sivil kayıplar
    20: 4,  # BİLİNMİYOR – Değerlendirilemediği için düşük ağırlık
    21: 6,  # ALTYAPI – Stratejik kesinti etkisi
    22: 7   # ŞİDDET EĞİLİMLİ POLİTİK PARTİLER – Politik olarak hassas hedefler
}

propextent_weights = {
    1: 10,  # Felaket boyutu (≥ 1 milyar $)
    2: 7,   # Büyük hasar (≥ 1 milyon < 1 milyar $)
    3: 4,   # Küçük hasar (< 1 milyon $)
    4: 5    # Bilinmiyor
}

# Normalleştirme fonksiyonu
def normalize_coefficient(coef_variable):
    return (coef_variable - 250) / (9000 - 250)

# Ağırlıklı milyon dolar cinsinden yıkıcılık indeksi hesaplama
def compute_index(row):
    print(normalize_coefficient(
        propextent_weights.get(row['propextent'], 0) *
        targtype1_weights.get(row['targtype1'], 0) *
        attack_type_1_weights.get(row['attacktype1'], 0) *
        weapon_weights.get(row['weaptype1'], 0)))
    
    return (1 if row["nkill"] != 0 else 0.1) * row["propvalue"] * normalize_coefficient(
        propextent_weights.get(row['propextent'], 0) *
        targtype1_weights.get(row['targtype1'], 0) *
        attack_type_1_weights.get(row['attacktype1'], 0) *
        weapon_weights.get(row['weaptype1'], 0)
    ) / (row["nkill"] if row["nkill"] != 0 else 1 / (targtype1_weights.get(row['targtype1'], 0) * 10))

# Filtreleme işlemi
df = df[df["propextent"].isin([1, 2, 3, 4])]
df = df[df["propvalue"] > 0]

# Yıkıcılık indeksini hesapla
df["DestructivenessIndex"] = df.apply(compute_index, axis=1)

# Sonuçları kaydet
df.to_csv("global_terrorism_dataset_filtered_desindex.csv", index=False)
