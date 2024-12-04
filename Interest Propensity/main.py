import random
import pandas as pd

attributes_encoded = pd.read_csv("/home/abp/Documents/Interest/attributes_encoded.csv")
final_df = pd.read_csv("/home/abp/Documents/Interest/final_df.csv")

# Create a list of all AWids
all_awids = attributes_encoded['AWid'].unique().tolist()

# Generate negative samples
num_negative_samples = len(final_df)  # Create an equal number of negative samples
negative_samples = []
for _ in range(num_negative_samples):
    source_awid = random.choice(all_awids)
    target_awid = random.choice(all_awids)
    # Ensure it's a negative sample (not in original final_df)
    while (source_awid, target_awid) in final_df[['SourceAwid', 'TargetAwid']].values.tolist():
        target_awid = random.choice(all_awids)
    negative_samples.append((source_awid, target_awid))

# Create a DataFrame for negative samples
negative_df = pd.DataFrame(negative_samples, columns=['SourceAwid', 'TargetAwid'])
negative_df['together'] = 0  # Label negative samples as 0

# Concatenate positive and negative samples
final_df_concat = pd.concat([final_df, negative_df], ignore_index=True)

# Saving 
final_df_concat.to_csv('final_df.csv', index=False)