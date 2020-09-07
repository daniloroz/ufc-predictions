import csv

with open("data.csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    next(reader, None)  #skip the headers
    data_read = [row for row in reader]

for i in range(6):
    print(data_read[i])