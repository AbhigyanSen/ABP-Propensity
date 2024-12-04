import pandas as pd

# Read the Attributes.csv only once to avoid multiple loads
attributes_df = pd.read_csv('Attributes.csv')

# Define the chunk size (adjust based on your available memory)
chunk_size = 10  # Number of rows to process at a time

# Initialize the writer for the output file
output_file = 'result.csv'
header_written = True

# Read Interest.csv in chunks and process
for chunk in pd.read_csv('Interest.csv', chunksize=chunk_size):
    # Merge the chunk with the Attributes.csv based on SourceAwid
    source_df = pd.merge(chunk, attributes_df, left_on='SourceAwid', right_on='AWid', how='left', suffixes=('_Source', '_Target'))
    
    # Merge the chunk with the Attributes.csv based on TargetAwid
    final_df = pd.merge(source_df, attributes_df, left_on='TargetAwid', right_on='AWid', how='left', suffixes=('_Source', '_Target'))

    # Select and reorder the necessary columns
    result_columns = [
        'SourceAwid', 'TargetAwid', 'createdOn', 'Communication_type',
       'userid_Source', 'AWid_Source', 'ProfileCreationDate_Source',
       'FirstPlanPurchaseDate_Source', 'Gender_Source', 'Age_Source',
       'City_Source', 'HomeTown_Source', 'Occupation_Source', 'Income_Source',
       'HighestEducation_Source', 'HighestEducationSpecialization_Source',
       'ProfileCreatedBy_Source', 'MaritalStatus_Source', 'Height_Source',
       'NoOfPhotosAdded_Source', 'UserStatus_Source',
       'PhysicalAppearance_Source', 'Complexion_Source', 'Disabilities_Source',
       'LivingHouseType_Source', 'familyIncome_Source', 'familyType_Source',
       'Caste_Source', 'Mangalik_Source', 'Unnamed: 25_Source',
       'Unnamed: 26_Source', 'Unnamed: 27_Source', 'Unnamed: 28_Source',
       'Unnamed: 29_Source', 'Unnamed: 30_Source', 'Unnamed: 31_Source',
       'Unnamed: 32_Source', 'userid_Target', 'AWid_Target',
       'ProfileCreationDate_Target', 'FirstPlanPurchaseDate_Target',
       'Gender_Target', 'Age_Target', 'City_Target', 'HomeTown_Target',
       'Occupation_Target', 'Income_Target', 'HighestEducation_Target',
       'HighestEducationSpecialization_Target', 'ProfileCreatedBy_Target',
       'MaritalStatus_Target', 'Height_Target', 'NoOfPhotosAdded_Target',
       'UserStatus_Target', 'PhysicalAppearance_Target', 'Complexion_Target',
       'Disabilities_Target', 'LivingHouseType_Target', 'familyIncome_Target',
       'familyType_Target', 'Caste_Target', 'Mangalik_Target',
       'Unnamed: 25_Target', 'Unnamed: 26_Target', 'Unnamed: 27_Target',
       'Unnamed: 28_Target', 'Unnamed: 29_Target', 'Unnamed: 30_Target',
       'Unnamed: 31_Target', 'Unnamed: 32_Target']
    
    # Write the result to the output file in batches
    # Open the file in append mode for subsequent chunks
    with open(output_file, 'a') as f:
        # Write the header for the first chunk
        if not header_written:
            final_df[result_columns].to_csv(f, index=False, header=True)
            header_written = True
        else:
            print("Final")
            print(final_df.columns)
            print("\n Result")
            print(result_columns)
            final_df[result_columns].to_csv(f, index=False, header=False)

    # Optional: Free memory by deleting intermediate dataframes if no longer needed
    del chunk, source_df, final_df