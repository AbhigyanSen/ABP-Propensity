import pandas as pd

df = pd.read_csv('city.csv')
df_cleaned = df.dropna()
df_cleaned.to_csv('city_cleaned.csv', index=False)
print("Null rows have been dropped and the cleaned file is saved as 'city_cleaned.csv'.")

df_cleaned = pd.read_csv('city_cleaned.csv')
unique_values = df_cleaned.iloc[:, 1].unique()
print("Unique values in the second column:")
print(unique_values)


# find city type using df["City"]
        df["CityType"] = "Rural"
        df = df.drop(["City"],axis=1)