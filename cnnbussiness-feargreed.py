import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

file_path = "fear-greed-2011-2023.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

#Aylık ortalamalar hesaplanır
monthly_avg = df.groupby(['Year', 'Month'])['Fear Greed'].mean().reset_index()



#Tarih verileri datetime nesnesine dönüştürülür
monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))

# Fear Kırmızı Greed Yeşil renginde olacak şekilde gradient renk geçişi hesaplaması
def get_gradient_color(value):
    normalized = value / 100
    red = 1 - normalized
    green = normalized
    return (red, green, 0)

# Renkleri hesapla
colors = [get_gradient_color(v) for v in monthly_avg['Fear Greed']]


plt.figure(figsize=(16, 6))
bars = plt.bar(monthly_avg['Date'], monthly_avg['Fear Greed'], width=20, color=colors)

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title("Monthly Fear & Greed Index with Gradient Colors (Red = Fear, Green = Greed)")
plt.xlabel("Year")
plt.ylabel("Fear & Greed Index")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
