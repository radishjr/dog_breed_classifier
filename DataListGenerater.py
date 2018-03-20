import pandas as pd
import os

current_path = os.path.dirname(os.path.abspath(__file__)) + "/"

print(current_path)
csv_path = current_path + "labels.csv"
dog_breed_df = pd.read_csv(csv_path)
updated = False
if not 'breed_int' in dog_breed_df:
    index_list = dog_breed_df['breed'].value_counts().index
    key_value_mapping = {}
    current_i = 0
    for index in index_list:
        key_value_mapping[index] = current_i
        current_i = current_i+1

    dog_breed_df['breed_int'] = dog_breed_df['breed'].map(key_value_mapping)
    updated = True

if not 'filepath' in dog_breed_df:
    dog_breed_df["filepath"] = dog_breed_df["id"].apply(lambda x: 
    current_path + "images/train/" + x + ".jpg")
    updated = True



if updated:
    dog_breed_df.to_csv(csv_path)