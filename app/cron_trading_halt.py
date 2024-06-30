import os
import sys
import pandas as pd



def main(method=""):
    print("Getting Trading Halt...")
    url = "https://www.nyse.com/api/trade-halts/current/download"
    df = pd.read_csv(url)
    df.fillna("N/A", inplace=True)
    df["Halt Date"] = df["Halt Date"].astype(str)
    df["Halt Date"] = df["Halt Date"].apply(lambda x: str(x[6:] + "-" + x[:2] + "-" + x[3:5] if x != "N/A" else "N/A"))
    df["Resume Date"] = df["Resume Date"].astype(str)
    #df["Resume Date"] = df["Resume Date"].apply(lambda x: str(x[6:] + "-" + x[:2] + "-" + x[3:5] if x != "N/A" else "N/A"))
    del df["Name"]
    print(df)
    print("Trading Halt Successfully Completed...\n")


if __name__ == '__main__':
    main()