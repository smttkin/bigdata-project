import os
import csv
from PyPDF2 import PdfReader

def pdf_istatistiklerini_topla(kok_dizin):
    istatistikler = []

    for alt_kok, _, dosyalar in os.walk(kok_dizin):
        pdf_sayisi = 0
        toplam_sayfa = 0
        toplam_karakter = 0

        for dosya in dosyalar:
            if dosya.lower().endswith(".pdf"):
                pdf_sayisi += 1
                pdf_yolu = os.path.join(alt_kok, dosya)
                try:
                    reader = PdfReader(pdf_yolu)
                    toplam_sayfa += len(reader.pages)
                    for sayfa in reader.pages:
                        text = sayfa.extract_text()
                        if text:
                            toplam_karakter += len(text)
                except Exception as e:
                    print(f"Hata: {pdf_yolu} dosyası okunamadı -> {e}")

        if pdf_sayisi > 0:
            istatistikler.append({
                "Klasör": os.path.relpath(alt_kok, kok_dizin),
                "PDF Sayısı": pdf_sayisi,
                "Toplam Sayfa Sayısı": toplam_sayfa,
                "Toplam Karakter Sayısı": toplam_karakter
            })

    return istatistikler

def csvye_yaz(istatistikler, cikti_dosyasi):
    with open(cikti_dosyasi, "w", newline="", encoding="utf-8") as csv_dosyasi:
        alanlar = ["Klasör", "PDF Sayısı", "Toplam Sayfa Sayısı", "Toplam Karakter Sayısı"]
        yazici = csv.DictWriter(csv_dosyasi, fieldnames=alanlar)
        yazici.writeheader()
        for satir in istatistikler:
            yazici.writerow(satir)

if __name__ == "__main__":
    klasor_yolu = os.getcwd()  # içinde bulunduğun klasör
    veriler = pdf_istatistiklerini_topla(klasor_yolu)
    csvye_yaz(veriler, "pdf_istatistikleri.csv")
    print("İstatistikler 'pdf_istatistikleri.csv' dosyasına kaydedildi.")
