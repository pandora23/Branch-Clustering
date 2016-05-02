

import csv

def getData(filename):

    data= []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|')
        attributeTags = reader.next()[0].split('\t')[:]
        data.append(attributeTags)
        print(attributeTags)
        for row in reader:
            data.append(row[0].split('\t')[:])
                        
    return data

#test                        
d1 =getData('taxonomySubtreeFreqs3levels2.csv')


