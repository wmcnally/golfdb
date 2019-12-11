from scipy.io import loadmat
import pandas as pd

x = loadmat('golfDB.mat')
l = list(x['golfDB'][0])
d = dict()
for idx, k in enumerate(l):
    d["{:3d}".format(idx)] = list(l[idx])
df = pd.DataFrame(d).T
df.columns = ["id","youtube_id","player", "sex", "club","view","slow","events","bbox","split"]

# data format cleansing
df['id'] = df['id'].apply(lambda x: x[0][0])
df['youtube_id'] = df['youtube_id'].apply(lambda x: x[0])
df['player'] = df['player'].apply(lambda x: x[0])
df['sex'] = df['sex'].apply(lambda x: x[0])
df['club'] = df['club'].apply(lambda x: x[0])
df['view'] = df['view'].apply(lambda x: x[0])
df['slow'] = df['slow'].apply(lambda x: x[0][0])
df['events'] = df['events'].apply(lambda x: x[0])
df['bbox'] = df['bbox'].apply(lambda x: x[0])
df['split'] = df['split'].apply(lambda x: x[0][0])

df.index = df.index.astype(int)
df.to_pickle('golfDB.pkl')

for i in range(1, 5):
    val_split = df.loc[df['split'] == i]
    val_split = val_split.reset_index()
    val_split = val_split.drop(columns=['index'])
    val_split.to_pickle("val_split_{:1d}.pkl".format(i))

    train_split = df.loc[df['split'] != i]
    train_split = train_split.reset_index()
    train_split = train_split.drop(columns=['index'])
    train_split.to_pickle("train_split_{:1d}.pkl".format(i))

print("Number of unique YouTube videos: {:3d}".format(len(df['youtube_id'].unique())))
print("Number of annotations: {:3d}".format(len(df.id)))