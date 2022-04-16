import sys
import csv
from pyproj import Transformer

if (len(sys.argv) < 3):
    print("Usage: python nadToNorm.py <input_file> <output_file>")
    exit(1)

transformer = Transformer.from_crs("epsg:2227", "epsg:4326")
temp = []

with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    temp.append(next(reader))
    for row in reader:
        converted = transformer.transform(row[3], row[2])
        row[3], row[2] = converted[0], converted[1]
        temp.append(row)

with open(sys.argv[2], 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(temp)
