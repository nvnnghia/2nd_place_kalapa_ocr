import os 
from glob import glob 
import pandas as pd

train_df = pd.read_csv('train_folds.csv')
fold_dict = {}
for i, row in train_df.iterrows():
    img_path = row['img_path']
    writer = img_path.split("/")[-2]
    name = img_path.split("/")[-1]
    fold = row['fold']
    fold_dict[f'{writer}_{name}'] = fold 

train_df['aug'] = 0 
train_df['img_path'] = 'training_data/images/' + train_df['img_path'] 

train_df['len'] = train_df['text'].apply(lambda x: len(x))
print(train_df['len'].max())

data_aug_paths = glob('synthetic_data/aug/*.txt')

ignore = [[46,0], [46,1], [67,2], [67,8], [67,9], [67,13], [67,19], [67,24], [68,-1],[73,1], [73,16], [73,17], [73,23], [73,24],
                        [76,3], [76,4], [76,5], [77,0], [77,1], [77,2], [77,3], [77,25], [87,4], [87,6],[87,8],[87,13],[87,21],
                        [87,26],[148,6],[148,22],[149,12],[167,-1],[195,-1]]

print(len(data_aug_paths))
results = []
for lb_path in data_aug_paths:
    with open(lb_path) as f:
        lines = f.readlines()

    if len(lines)>1:
        print(lines)

    lines = lines[0]

    # tes = line.split()
    file_name = lb_path.replace('txt', 'jpg')
    if os.path.isfile(file_name):
        text = lines.strip()
        name = file_name.split('/')[-1]
        writer = name.split('_')[0]
        name = name.split('_')[1]

        is_skip = False
        for ig in ignore:
            if int(writer) == ig[0]:
                if int(name[:-4]) == ig[1] or ig[1] == -1:
                    is_skip = True 
        if is_skip:
            # print(file_name)
            continue

        fold = fold_dict[f'{writer}_{name}']
        results.append({'img_path': file_name, 'text': text, 'writer_id': 0, 'fold': fold, 'aug': 1})
    # print(file_name, text)
    # break
# break

df1 = pd.DataFrame(results)
print(df1.shape)

print('====== process v1====')
### gendata v1
data_v1_paths = glob('synthetic_data/address/*.txt')
print(len(data_v1_paths))
results = []
for lb_path in data_v1_paths:
    with open(lb_path) as f:
        lines = f.readlines()

    if len(lines)>1:
        print(lines)

    lines = lines[0]

    # tes = line.split()
    file_name = lb_path.replace('txt', 'jpg')
    if os.path.isfile(file_name):
        text = lines.strip()
        name = file_name.split('/')[-1]
        idx = name.split('_')[0]
        font = name.split('_')[1].split('.')[0]
        # print(font)
        if font == 'Babylonica-Regular':
            # print(file_name)
            continue  
        if font == 'propaniac':
            text = text.upper()
            # print(text)
        aug = 0
        if int(idx) > 50000:
            aug = 1
        fold = 6
        results.append({'img_path': file_name, 'text': text, 'writer_id': 0, 'fold': fold, 'aug': aug})
    # print(file_name, text)
    # break
# break

df2 = pd.DataFrame(results)
print(df2.shape)

print('====== process text ====')

### gendata text
data_text_paths = glob('synthetic_data/poems/*.txt')
print(len(data_text_paths))
results = []
for lb_path in data_text_paths:
    with open(lb_path) as f:
        lines = f.readlines()

    if len(lines)>1:
        print(lines)

    lines = lines[0]

    # tes = line.split()
    file_name = lb_path.replace('txt', 'jpg')
    if os.path.isfile(file_name):
        text = lines.strip()
        name = file_name.split('/')[-1]
        idx = name.split('_')[0]
        aug = name.split('_')[1]
        font = name.split('_')[2].split('.')[0]

        if font == 'Babylonica-Regular':
            # print(file_name)
            continue

        if font == 'propaniac':
            text = text.upper()
            # print(text)
        aug = int(aug)
        fold = 6
        results.append({'img_path': file_name, 'text': text, 'writer_id': 0, 'fold': fold, 'aug': aug})
    # print(file_name, text)
    # break
# break

df3 = pd.DataFrame(results)
print(df3.shape)

df = pd.concat([train_df,df1,df2,df3]).sample(frac=1.0)

df['len'] = df['text'].apply(lambda x: len(x))
print(df['len'].max())
df = df[df['len']<70]
print(df.shape)
# print(df.head())
df.to_csv('train_ext3.csv',index=False)