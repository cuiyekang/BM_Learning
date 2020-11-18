
import pandas as pd

data = pd.read_excel("./docs/python/t_data/神奇公式数据.xlsx",skiprows=4,sheet_name="2017")

data = data[(data["pe_de"]>0) & (data["roic"]>0) & (data["pe_year1"]>0) & (data["pe_year2"]>0) & (data["roic_year1"]>0) & (data["roic_year2"]>0)]

rank_pe = data['pe_de'].rank(method='average')
rank_roic = data['roic'].rank(method='average',ascending=False)

rank_used = rank_pe + rank_roic
rank_used = pd.DataFrame(rank_used).rename(columns = {0:'rank'})

new_data = pd.merge(data,rank_used,left_index=True,right_index=True)
new_data.sort_values(by='rank',inplace=True)
new_data.reset_index(drop=True,inplace=True)

final_choice = new_data.iloc[:30,:]
final_codes = list(final_choice["Wind代码"].values)
print(final_codes)