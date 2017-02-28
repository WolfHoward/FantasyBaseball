
# coding: utf-8

# In[1]:

import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA


# Assigning methods to easily grab record data for display or calculation

class Record(object):

    def __init__(self, df):
        self.df = hit_summary(df)
        self.obp = self.df['OBP']
        self.slg = self.df['SLG']
        self.ops = self.df['OPS']
        self.avg = self.df['AVG']



# On Base Percentage
#       (H + BB + HBP) / (AB + BB + SF + HBP)
#       As calculated on baseball-reference.com

def OBP(df):
    AB, H, BB, HBP, SF = df['AB'], df['H'], df['BB'], df['HBP'], df['SF']
    obp = (H + BB + HBP)/(AB + BB + SF + HBP)
    return(obp)


# Slugging Percentage
#       (H + 2B + 3B*2 + HR*3) / (AB)
#       As calculated on baseball-reference.com, adjusted to avoid calculating
#       singles explicitly.

def SLG(df):
    AB, H, B2, B3, HR = df['AB'], df['H'], df['2B'], df['3B'], df['HR']
    tb = H + B2 + 2*B3 + 3*HR
    slg = tb/AB
    return(slg)


# On Base Plus Slugging

def OPS(df):
    OBP, SLG = df['OBP'], df['SLG']
    ops = OBP + SLG
    return(ops)


# Batting Average

def AVG(df):
    H, AB = df['H'], df['AB']
    avg = H/AB
    return(avg)


# Prints a summary of hitting statistics, running the stat calculations above.
# If sgp == True, run a standings gain points calculation.

def hit_summary(df,sgp=False):
    df['AVG'] = AVG(df)
    df['OBP'] = OBP(df)
    df['SLG'] = SLG(df)
    df['OPS'] = OPS(df)

    if sgp == True:
        df = sgp_calc(df)

    return(df)


# Calculate standings gain points based on league settings and given league
# data

def sgp_calc(df, teams=10, hitters=9):
    players = teams * hitters
    top_hitters = df.iloc[players]
    ab_mean = top_hitters['AB'].mean()
    ab_team = at_bats * teams * hitters


    #.27


# In[ ]:
'''
def kmeans_example(df)
    df = df.dropna(axis=0)
    kmeans_model = KMeans(n_clusters=5, random_state=1)
    good_columns = df._get_numeric_data()
    kmeans_model.fit(good_columns)
    labels = kmeans_model.labels_

    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(good_columns)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
    plt.show()

#    projected_average = df['AVG'].mean()
'''

# Read in the Lahman database. Current up to 2015. We will append 2016 data via
# a separate web-scraping method.


teams = pd.read_csv('lahman-2015/Teams.csv')


# In[44]:

hitting = pd.read_csv('lahman-2015/batting.csv')
player_map = pd.read_excel('SFBB-Player-ID-Map.xlsx')
hit_proj = pd.read_csv('SteamerHitters.csv')
hit_proj.head()
hit_proj = hit_proj[['Name','Team','PA', 'AB', 'H', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'SO', 'HBP', 'SB']]


# In[45]:

hit_proj['SF'] = hit_proj['PA'] - hit_proj['AB'] - hit_proj['HBP'] - hit_proj['BB']
hit_proj = hit_summary(hit_proj)


# In[46]:

hit_rank = hitting[hitting['yearID'] >= 2008]
hit_rank = hit_rank[hit_rank['AB'] >= 10]
hit_rank = hit_summary(hit_rank)


mlb_annual_stats = hit_rank.groupby(['yearID']).sum()
player_stats = hitting[hitting['yearID'] >= 2015]
player_stats = player_stats.groupby(['playerID'])

mlb_annual_stats.sort_index(ascending=False, inplace=True)

mlb_annual_stats = hit_summary(mlb_annual_stats)
mlb_class = Record(mlb_annual_stats)
mlb_annual_stats.head()



ab_std = mlb_annual_stats.mean(axis=0)
print(ab_std)



# In[53]:

plt.figure(2)
plt.subplot(311)
colors = ('k', 'r', 'b')
x = mlb_annual_stats.index.values
y1 = mlb_annual_stats['H']
y2 = mlb_annual_stats['OBP']
y3 = mlb_annual_stats['SLG']
plt.plot(x, y1)
plt.subplot(312)
plt.plot(x, y2)
plt.subplot(313)
plt.plot(x, y3)
plt.show()


# In[54]:

hit_proj.head()


# In[55]:

sgp_calc(hitting[hitting['yearID'] >= 2015])

'''
# In[56]:

columns = hit_rank.columns.tolist()
columns = [c for c in columns if c in ["AB", "R", "H", "2B", '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'HBP', 'SF']]
print(columns)
target = "H"
from sklearn.cross_validation import train_test_split


# In[57]:

train = hit_rank.sample(frac=0.8, random_state=1)
test = hit_rank.loc[~hit_rank.index.isin(train.index)]

print(train.shape)
print(test.shape)


# In[58]:

from sklearn.linear_model import LinearRegression
X=train[columns].drop('H', axis=1)
Y = train[target]
#print(X)
model = LinearRegression()
model.fit(X, Y)
#print(model.intercept_, len(model.coef_))


# In[59]:

pd.DataFrame(list(zip(train[columns], model.coef_)), columns = ['features', 'estimatedCoefficients'])


# In[60]:

plt.scatter(hit_rank['H'], hit_rank['2B'])
plt.show()


# In[62]:

Y = model.predict(X)
print(X.shape, Y.shape)
plt.scatter(X, Y)
'''

# In[ ]:

print(mlb_annual_stats['AVG'])
