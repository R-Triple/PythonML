import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits


#import the data set
data = pd.read_csv("kc_house_data.csv")

# read the datas first lines
data.head()

# read the datas last lines
data.tail()

# see the range of the different house builds and gives us a starting point to
# infering the picture of the data set. For example, we can see that the biggest
# house has 33 bedrooms.
data.describe()

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine

