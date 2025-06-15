import yfinance as yf

tickers=["^N225","XU100.IS","BRAX11.SA","^SP100","IMOEX.ME","3988.HK"]
for ticker in tickers:
    currticker = yf.download(ticker, start="2007-01-01", end="2025-01-01",interval="1mo")
    currticker = currticker.resample('Y').last()  

    # Veriyi göster
    print(currticker.head())


    currticker.to_csv(f"OHLCV/{ticker}.csv",index=True)