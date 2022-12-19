import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import random


def main():
    moves = []
    test_moves = []
    food = []
    with open('training_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if len(row[0]) != 30:
                continue
            if row[1] == '0':
                moves.append(row[0])
                food.append(row[1])
            elif row[1] == '1':
                moves.append(row[0])
                food.append(row[1])
            else:
                continue

    with open('training_data1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if len(row[0]) != 30:
                continue
            if row[1] == '0':
                moves.append(row[0])
                food.append(row[1])
            elif row[1] == '1':
                moves.append(row[0])
                food.append(row[1])
            else:
                continue

    with open('training_data2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if len(row[0]) != 30:
                continue
            if row[1] == '0':
                moves.append(row[0])
                food.append(row[1])
            elif row[1] == '1':
                moves.append(row[0])
                food.append(row[1])
            else:
                continue

    with open("simple_test_data.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test_moves.append(row[0])

    random.shuffle(moves)
    random.shuffle(test_moves)
    model = DecisionTreeClassifier()
    le = preprocessing.LabelEncoder()
    le.fit(moves)
    moves_list = le.transform(moves)
    model.fit(moves_list.reshape(-1, 1), food)

    le.fit(test_moves)
    test_moves_list = le.transform(test_moves)
    predictions = model.predict(test_moves_list.reshape(-1, 1))
    print(predictions)
    with open("out.txt", "w") as text_file:
        for i in range(len(predictions)):
            text_file.write('%s\n' % predictions[i])


if __name__ == "__main__":
    main()
