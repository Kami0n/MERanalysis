
from os import listdir
from os.path import isfile, join
import os
import csv


folder = 'Dataset/'
mypath = folder+'audio'


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = {x.replace('.mp3', '') for x in onlyfiles}
#print(onlyfiles)

mypath = folder+'metadata/'

csvFileNames = ['metadata_2013', 'metadata_2014', 'metadata_2015']


allGeneres = []
for name in csvFileNames:
    inputPath = mypath+name+'.csv'
    
    with open(inputPath, newline='', encoding='utf-8') as csvfile:
        fileReader = csv.reader(csvfile, delimiter=',')
        
        header = next(fileReader)
        headerNew = [item.lower() for item in header]
        indexGenre = headerNew.index('genre')
        #print(indexGenre)
        
        for row in fileReader:
            if row[0] in onlyfiles:
                genere = ' '.join(row[indexGenre].lower().split()) # strip whitespace and to lowercase
                allGeneres.append([row[0],genere])
                
print(allGeneres)

myset = set(allGeneres)
print(myset)



fields = ['song_id','genere']

with open(mypath+'allGeneres.csv', 'w', newline='', encoding='utf-8') as f: 
    
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
    
    write.writerow(fields) 
    write.writerows(allGeneres) 