import os
import numpy as np
import pandas as pd

data_filename = "./docs/python/t_data/matches.csv"

#result = pd.read_csv(data_filename,parse_dates=["Date"],skiprows=[0,])
results = pd.read_csv(data_filename,skiprows=[0,])
results.columns=["RowId","Date", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]

# results[]

removeIndex = 0
for index, row in results.iterrows():
    if str(row["Visitor Team"]) == "nan":
        removeIndex = index


results = results.drop(index=[removeIndex])
results.reset_index(drop=True,inplace=True)

print(results.iloc[1317:1319])


#print(results[np.isnan(results["Visitor Team"].values)])

# results[results["Visitor Team"]!=""]

# print(results.count())
