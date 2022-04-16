import sys
import csv
import re
points = []


def chopMile(milepost):
    number = milepost[1:len(milepost)]
    return float(number)


if len(sys.argv) != 2:
    sys.exit("Usage: python csvToUnderscore.py <filename>")

with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        points.append(row)

LineA = []
LineC = []
LineM = []
LineR = []
for row in points:
    milepost = row[5]
    line = milepost[0]
    if line == 'A':
        LineA.append(row)
    elif line == 'C':
        LineC.append(row)
    elif line == 'M':
        LineM.append(row)
    elif line == 'R':
        LineR.append(row)
    else:
        LineA.append(row)

Lines = [LineA,LineC,LineM,LineR]
finalPoints = []
for line in Lines:
    aggPoints = []
    for row in line:
        found = False
        mileNumber = chopMile(row[5])
        for point in aggPoints:
            dist = abs(mileNumber-chopMile(point[5]))
            if dist < 1:
                bearing = abs(float(row[4])-float(point[4]))
                if bearing <10:
                    found = True
        if not found:
            aggPoints.append(row)
    for point in aggPoints:
        finalPoints.append(point)

with open("output.csv", 'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerows(finalPoints)