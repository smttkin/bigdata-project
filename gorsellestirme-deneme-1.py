import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates

# 1. Veri setleri okunur 
df1 = pd.read_csv("global_terrorism_dataset_filtered_desindex.csv")   # iyear, DestructivenessIndex
df2 = pd.read_csv("sentiment-v4.csv") # Year, Company, Polarity, Subjectivity




def calculate_polarity(pos, neg, neu):
    return (pos - neg) / (pos + neg + neu + 1e-6)  # VADER tarzı normalize edilmiş polarite
df2["Polarity"] = df2.apply(lambda row: calculate_polarity(row["Positive"], row["Negative"], row["Neutral"]), axis=1)


#Polarity hesabı yapılmış veri seti kaydedilir.
df2.to_csv("sentiment-v4.csv", index=False)

df2['Year'] = df2['Year'].astype(int)

# Yıl bazlı ortalama Destructiveness Index hesaplanır
des_avg = df1.groupby('iyear')['DestructivenessIndex'].mean().reset_index()

des_avg.rename(columns={'iyear': 'Year', 'DestructivenessIndex': 'AvgDestructiveness'}, inplace=True)
des_avg['Year'] = des_avg['Year'].astype(int)
des_avg = des_avg[des_avg['Year'] >= 1997]

# İkinci verisetinde hesaplanan ortalama DestructivenessIndex değerleri eklenir. 
df2 = df2.merge(des_avg, on='Year', how='left')

# -- FearGreed için hazırlık
df3 = pd.read_csv("fear-greed-2011-2023.csv")
df3['Date'] = pd.to_datetime(df3['Date'])
df3['Year'] = df3['Date'].dt.year
df3['Month'] = df3['Date'].dt.month
monthly_avg = df3.groupby(['Year', 'Month'])['Fear Greed'].mean().reset_index()
monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))

def get_gradient_color(value):
    normalized = value / 100
    red = 1 - normalized
    green = normalized
    return (red, green, 0)

colors = [get_gradient_color(v) for v in monthly_avg['Fear Greed']]

# --- X EKSENİ AYARLARI İÇİN: datetime tipinde Year sütunları oluştur
df2['YearDate'] = pd.to_datetime(df2['Year'].astype(str) + '-01-01')
des_avg['YearDate'] = pd.to_datetime(des_avg['Year'].astype(str) + '-01-01')

# X ekseninin min-max sınırlarını belirle (en erken ve en geç tarih)
min_year = min(df2['YearDate'].min(), des_avg['YearDate'].min(), monthly_avg['Date'].min())
max_year = max(df2['YearDate'].max(), des_avg['YearDate'].max(), monthly_avg['Date'].max())

min_index = des_avg['AvgDestructiveness'].min()
max_index =  des_avg['AvgDestructiveness'].max()

plt.figure(figsize=(16, 18))

plt.subplot(4,1,1)
sns.lineplot(data=df2, x='YearDate', y='Polarity', hue='Company', marker='o')
plt.title("Şirketlere Göre Polarity Değişimi")
plt.ylabel("Polarity")
plt.xlim(min_year, max_year)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.subplot(4,1,2)
sns.lineplot(data=df2, x='YearDate', y='Subjectivity', hue='Company', marker='o', palette='tab10')
plt.title("Şirketlere Göre Subjectivity Değişimi")
plt.ylabel("Subjectivity")
plt.xlim(min_year, max_year)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.subplot(4,1,3)
sns.lineplot(data=des_avg, x='YearDate', y='AvgDestructiveness', color='black', linewidth=2)
plt.title("Yıllık Ortalama Destructiveness Index (1997 ve sonrası)")
plt.xlabel("Yıl")
plt.ylabel("Ortalama Destructiveness")
plt.xlim(min_year, max_year)

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


plt.subplot(4,1,4)
bars = plt.bar(monthly_avg['Date'], monthly_avg['Fear Greed'], width=20, color=colors)
plt.title("Aylık Fear & Greed Index (Gradient Renkli)")
plt.ylabel("Fear & Greed Index")
plt.ylim(0, 100)
plt.xlim(min_year, max_year)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xlabel("Yıl")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Seaborn Heatmap (Polarity)
pivot = df2.pivot_table(index='Company', columns='Year', values='Polarity')

plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt=".2f", cbar_kws={'label': 'Polarity'})
plt.title("Şirketlere Göre Yıllık Polarity Isı Haritası")
plt.xlabel("Yıl")
plt.ylabel("Şirket")
plt.show()

# Plotly — Polarity, Subjectivity ve AvgDestructiveness Hover ile
fig = px.line(df2, x='Year', y='Polarity', color='Company', markers=True,
              title="Şirketlere Göre Yıllık Polarity ve Hover'da Subjectivity & Destructiveness",
              hover_data={'AvgDestructiveness': ':.2f', 'Subjectivity': ':.2f'})

fig.update_layout(xaxis_title='Yıl', yaxis_title='Polarity')
fig.show()
