import math

# 1. Read the original dataset file
with open('bank-full.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract header and data rows
header = lines[0]
rows = lines[1:]

# 2. Calculate size for each of the 3 parts
chunk = math.ceil(len(rows) / 3)

# 3. Divide rows into 3 portions
part1 = rows[:chunk]
part2 = rows[chunk:2*chunk]
part3 = rows[2*chunk:]

# 4. Save each portion into a new file with the header included
with open('bank_master.csv', 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(part1)

with open('bank_worker1.csv', 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(part2)

with open('bank_worker2.csv', 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(part3)

print("Dataset successfully split into 3 parts: bank_master.csv, bank_worker1.csv, bank_worker2.csv")