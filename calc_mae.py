import csv

with open('output_new.csv', 'r') as t1, open('truth.csv', 'r') as t2:
    sum = 0
    count = 0
    fileone = t1.readlines()
    filetwo = t2.readlines()

    for row1, row2 in zip(fileone, filetwo):
        v1 = row1.split(',')
        v2 = row2.split(',')
        sum += abs((float(v1[1]) - float(v2[1])))
        count += 1

    mae = sum / count
    print(mae)

with open('output_new.csv', 'r') as t1, open('truth.csv', 'r') as t2:
    sum = 0
    count = 0
    fileone = t1.readlines()
    filetwo = t2.readlines()

    for row1, row2 in zip(fileone, filetwo):
        v1 = row1.split(',')
        v2 = row2.split(',')
        sum += (float(v1[1]) - float(v2[1])) * (float(v1[1]) - float(v2[1]))
        count += 1

    rmse = sum ** 0.5
    rmse = rmse / (count ** 0.5)

    print(rmse)

