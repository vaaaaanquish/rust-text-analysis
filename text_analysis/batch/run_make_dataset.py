import random
import os
import re
import glob


reg_ignore = re.compile('license', re.IGNORECASE)
files = os.path.join('text', '**', '*.txt')
data = []

for f in glob.glob(files):
    if reg_ignore.search(os.path.basename(f)):
        continue

    d, _ = os.path.split(f)
    media = os.path.basename(d)
    with open(f, 'r', encoding='utf-8', newline='') as rf:
        for _ in range(3):
            # url date title
            _ = next(rf)
        body = ''.join(line.strip() for line in rf if line.strip())
        data.append(f'{media}\t{body}')


index = list(range(len(data)))
sample_size = int(len(data) * 0.3)
test_index = random.sample(index, sample_size)
train_index = [x for x in index if x not in test_index]

test_data = [data[i] for i in test_index]
train_data = [data[i] for i in train_index]

labels = list(set([x.split('\t')[0] for x in train_data]))

with open('/app/data/train.csv', 'w') as f:
    f.write('text,label\n')
    for x in train_data:
        row = x.split('\t')
        label = labels.index(row[0])
        text = ''.join(row[1].replace(',', '').replace('\n', ' ').split(' '))
        f.write(f'{text},{label}\n')


with open('/app/data/test.csv', 'w') as f:
    f.write('text,label\n')
    for x in test_data:
        row = x.split('\t')
        label = labels.index(row[0])
        text = ''.join(row[1].replace(',', '').replace('\n', ' ').split(' '))
        f.write(f'{text},{label}\n')
