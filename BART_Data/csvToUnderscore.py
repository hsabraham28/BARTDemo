"""
Converts first column (ObjectID) of CSV files to underscore syntax
Replaces all whitespaces, commas, dashes, to underscores

Usage: python csvToUnderscore.py <filename>
"""
import sys
import csv
import re

temp = []

if len(sys.argv) != 2:
    sys.exit("Usage: python csvToUnderscore.py <filename>")

with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row[0] = re.sub('[,.?!\t\n ]+', '_', row[0]).strip('_')
        temp.append(row)

with open(sys.argv[1], 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(temp)
