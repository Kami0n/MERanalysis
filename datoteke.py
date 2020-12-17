
from os import listdir
from os.path import isfile, join
import os
import csv

dirname = os.path.dirname(__file__)
mypath = os.path.join(dirname, 'Dataset/audio')

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = {x.replace('.mp3', '') for x in onlyfiles}
#print(onlyfiles)

filteredEmotions = []
inputPath = os.path.join(dirname, 'Dataset/emotions/static_annotations_averaged_songs_1_2000.csv')
with open(inputPath, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] in onlyfiles:
            filteredEmotions.append(row)


inputPath = os.path.join(dirname, 'Dataset/emotions/static_annotations_averaged_songs_2000_2058.csv')

included_cols = [0, 1, 2, 7, 8]

with open(inputPath, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        content = list(row[i] for i in included_cols)
        filteredEmotions.append(content)

print(filteredEmotions)

fields = ['song_id','valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']
outputPath = os.path.join(dirname, 'Dataset/emotions/filtered_annotations_vse.csv')
with open(outputPath, 'w', newline='', encoding='utf-8') as f: 
    
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
    
    write.writerow(fields) 
    write.writerows(filteredEmotions) 