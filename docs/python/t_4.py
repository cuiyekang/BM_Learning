import os
import numpy as np
import pandas as pd

data_filename = "./docs/python/t_data/matches.csv"

#result = pd.read_csv(data_filename,parse_dates=["Date"],skiprows=[0,])
results = pd.read_csv(data_filename,skiprows=[0,])
results.columns=["RowId","Date", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]

results["HomeWin"]=results["HomePts"]>results["VisitorPts"]



y_true=results["HomeWin"].values

print(results[:5])

print("Home Win percentage: {0:.1f}%".format(100 * results["HomeWin"].sum() / results["HomeWin"].count()))

results["HomeLastWin"]=False
results["VisitorLastWin"]=False

from collections import defaultdict

won_last=defaultdict(int)

for index,row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team=row["Visitor Team"]

    row["HomeLastWin"]=won_last[home_team]
    row["VisitorLastWin"]=won_last[visitor_team]

    results.iloc[index]=row

    won_last[home_team]=row["HomeWin"]
    won_last[visitor_team]=not row["HomeWin"]

print(results[20:25])


##########单队伍上次输赢

# from sklearn.tree import DecisionTreeClassifier

# clf=DecisionTreeClassifier(random_state = 14)

# from sklearn.model_selection import cross_val_score

# x_previouswins=results[["HomeLastWin","VisitorLastWin"]].values

# scores=cross_val_score(clf,x_previouswins,y_true,scoring='accuracy')

# print("Using just the last result from the home and visitor teams")
# print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))



#######主客场输赢

# results["HomeWinStreak"]=0
# results["VisitorWinStreak"]=0

# win_streak=defaultdict(int)

# for index,row in results.iterrows():
#     home_team = row["Home Team"]
#     visitor_team = row["Visitor Team"]
#     row["HomeWinStreak"]=win_streak[home_team]
#     row["VisitorWinStreak"]=win_streak[visitor_team]
#     results.iloc[index]=row
#     if row["HomeWin"]:
#         win_streak[home_team]+=1
#         win_streak[visitor_team]=0
#     else:
#         win_streak[home_team]=0
#         win_streak[visitor_team]+=1

# print(results[20:25])

# from sklearn.tree import DecisionTreeClassifier

# clf=DecisionTreeClassifier(random_state=14)

# x_winstreak=results[["HomeLastWin","VisitorLastWin","HomeWinStreak","VisitorWinStreak"]].values

# from sklearn.model_selection import cross_val_score

# scores=cross_val_score(clf,x_winstreak,y_true,scoring="accuracy")

# print("Using whether the home team is ranked higher")
# print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


#####

removeIndex = 0
for index, row in results.iterrows():
    if str(row["Visitor Team"]) == "nan":
        removeIndex = index
y_true=np.delete(y_true,removeIndex,axis=0)
results = results.drop(index=[removeIndex])
results.reset_index(drop=True,inplace=True)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoding = LabelEncoder()
encoding.fit(results["Home Team"].values)
home_teams=encoding.transform(results["Home Team"].values)
visitor_teams = encoding.transform(results["Visitor Team"].values)

x_teams = np.vstack([home_teams,visitor_teams]).T

onehot = OneHotEncoder()
x_teams=onehot.fit_transform(x_teams).todense()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf=DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf,x_teams,y_true,scoring="accuracy")

print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state = 14)
scores=cross_val_score(clf,x_teams,y_true,scoring="accuracy")
print("Using full team labels is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


