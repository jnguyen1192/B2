import unittest

import sklearn
import csv


class TestClustering(unittest.TestCase):

    def test_prepare_dataset(self):
        def readCsv(file, header=False, delimiter=',', newline='\n'):
            """
            Read the CSV file with the hearder or not
            :param file: the path of the file
            :param header: if there was a header on the file
            :param delimiter: the delimter between each columns
            :param delimiter: the delimter between each columns
            :return: the array containing the values of the dataset
            """
            if not header:
                c = 0
            else:
                c = 1
            with open(file, newline=newline) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=delimiter)
                for row in spamreader:
                    if c == 0:
                        c = 1
                        continue
                    yield row

        data = readCsv('dataset/buddymove_holidayiq.csv', True)
        for d in data:
            print(d)


if __name__ == '__main__':
    unittest.main()