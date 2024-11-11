import csv

# Assuming the CSV file is named 'data.csv'
data = []

# Open the CSV file and read its content
with open('Dropdown.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

# Extracting values for each column, ignoring blank or missing values
Age = [row['Age'] for row in data if row['Age'].strip() != '']
Gender = [row['Gender'] for row in data if row['Gender'].strip() != '']
City = [row['City'] for row in data if row['City'].strip() != '']
Occupation = [row['Occupation'] for row in data if row['Occupation'].strip() != '']
Industry = [row['Industry'] for row in data if row['Industry'].strip() != '']
Income = [row['Income'] for row in data if row['Income'].strip() != '']
Education = [row['Education'] for row in data if row['Education'].strip() != '']
Specialisation = [row['Specialisation'] for row in data if row['Specialisation'].strip() != '']
Profile_Created_by = [row['Profile Created by'] for row in data if row['Profile Created by'].strip() != '']
Marital_status = [row['Marital status'] for row in data if row['Marital status'].strip() != '']
Height = [row['Height'] for row in data if row['Height'].strip() != '']

# Now print the lists to verify
print("Age:", Age)
print("Gender:", Gender)
print("City:", City)
print("Occupation:", Occupation)
print("Industry:", Industry)
print("Income:", Income)
print("Education:", Education)
print("Specialisation:", Specialisation)
print("Profile Created by:", Profile_Created_by)
print("Marital status:", Marital_status)
print("Height:", Height)