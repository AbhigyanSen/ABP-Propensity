import os
import pandas as pd

for filename in os.listdir("/home/abp/Documents/Interest/ABPData"):
  df = pd.read_csv(f"/home/abp/Documents/Interest/ABPData/{filename}")
  print(f"{filename}: {df.shape}")
  null_percentage_profileViews = df.isnull().mean() * 100
  print("Percentage of null values in each column:")
  print(null_percentage_profileViews)
  print("\n")
  df = df.fillna("AWdel")
  df = df[~df.apply(lambda row: row.astype(str).str.contains('AWdel').any(), axis=1)]

  print("\n")
  null_percentage_profileViewssp = df.isnull().mean() * 100
  print("Percentage of null values after processing:")
  print(null_percentage_profileViewssp)
  df.to_csv(f"/home/abp/Documents/Interest/ABPData/{filename}", index=False)