import pandas as pd

def ff_monthly() -> pd.DataFrame:
    a = pd.read_csv("data/raw/french/F-F_Research_Data_5_Factors_2x3.csv", skiprows=3)
    a = a[a.iloc[:, 0].astype(str).str.strip().str.len() == 6].rename(columns={a.columns[0]: "ym"})
    a["date"] = pd.to_datetime(a.ym.str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    a = a.set_index("date").drop(columns="ym").astype(float) / 100
    m = pd.read_csv("data/raw/french/F-F_Momentum_Factor.csv", skiprows=13)
    m = m[m.iloc[:, 0].astype(str).str.strip().str.len() == 6].rename(columns={m.columns[0]: "ym"})
    m["date"] = pd.to_datetime(m.ym.str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    m = m.set_index("date").drop(columns="ym").astype(float) / 100
    m.columns = ["MOM"]
    return a.join(m, how="inner")
