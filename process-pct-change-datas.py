import pandas as pd
import os


#İlgili borsa OHLCV  verilerinden Close Değeri Yüzdelik Değişimleri hesaplanır

folder_path = './OHLCV/' 

all_yearly_data = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        full_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(full_path, skiprows=2)
        print(df.columns,df.head())
        df.rename(columns={
        'Unnamed: 1': 'Close',
        'Unnamed: 2': 'High',
        'Unnamed: 3': 'Low',
        'Unnamed: 4': 'Open',
        'Unnamed: 5': 'Volume',
        },inplace=True)


        # Date sütununu datetime yap
        df['Date'] = pd.to_datetime(df['Date'])

        # Tarihi index yap
        df.set_index('Date', inplace=True)

        # Yıllık gruplama
        df['Year'] = df.index.year

        yearly = df.groupby('Year').agg({
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'
        })

        yearly['close_yoy_pct'] = yearly['Close'].pct_change() * 100
        yearly['range_pct'] = ((yearly['High'] - yearly['Low']) / yearly['Open']) * 100
        yearly['volume_yoy_pct'] = yearly['Volume'].pct_change() * 100

        yearly = yearly.dropna().reset_index()

        symbol = file_name.replace('.csv', '')
        yearly['symbol'] = symbol

        yearly = yearly[['Year', 'symbol', 'close_yoy_pct', 'range_pct', 'volume_yoy_pct']]
        yearly.rename(columns={'Year': 'date'}, inplace=True)

        all_yearly_data.append(yearly)

final_df = pd.concat(all_yearly_data, ignore_index=True)

print(final_df.head())
final_df.to_csv("pct_change.csv",index=True)   