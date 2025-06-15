import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates

df1 = pd.read_csv("global_terrorism_dataset_filtered_desindex.csv") 
avg_destruction_by_year = df1.groupby('iyear')['DestructivenessIndex'].mean()

yearly_attacks = df1['iyear'].value_counts().sort_index()
plt.figure(figsize=(12,6))
plt.plot(yearly_attacks.index, avg_destruction_by_year.values,marker="o")
plt.title("Yıllara Göre Terör Saldırı Sayısı")
plt.xlabel("Yıl")
plt.ylabel("Saldırı Sayısı")
plt.grid(True)
plt.show()