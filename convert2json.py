import json
import pandas as pd
import re

df = pd.read_csv('inference_results.csv')
df['ID'] = df['ID'].str.extract(r'(\d+)').astype(int)
df = df.sort_values(by=['ID'], ascending=True)

with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df_index = 0
for item in data["RECORDS"]:
    if df_index >= len(df):
        break
    data_index = int(item["hashcode"])
    while df['ID'].iloc[df_index] < data_index:
        df_index += 1
        if df_index >= len(df):
            break
    if df_index >= len(df):
        break
    if df['ID'].iloc[df_index] > data_index:
        item['Result'] = 'None'
        continue
    elif df['ID'].iloc[df_index] == data_index:
        result = re.sub(r"^\['(.*)'\]$", r"\1", df['Result'].iloc[df_index])
        item['Result'] = result
        df_index += 1
with open('data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)