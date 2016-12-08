#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import numpy as np
# import pandas as pd
# import seaborn as sns
#
# x = np.random.normal(size=100) #ランダムデータをnumpy arrayとして作る
#
# titanic = sns.load_dataset("titanic") ##kaggleで有名な、タイタニック号の生死者データ
# tips = sns.load_dataset("tips")  ## お店の食事時間と会計総額とチップの関係のデータ
# iris = sns.load_dataset("iris")  ## Rでお馴染みのアヤメの統計データ
# print(titanic)





import seaborn as sns
import pandas as pd

sns.set()
df = sns.load_dataset("iris")
df = pd.DataFrame([[1,None,None, 'a'],[None,3,4,'b'],[None,None,None,'c']], columns=['a','b','c','d'])
sns.pairplot(df, hue='d', dropna=True)
print(df)
sns.plt.show()

# import matplotlib.pyplot as plt
# from numpy.random import randn
# z = randn(10)
# green_dot, = plt.plot(z, "go", markersize=3)
# red_dot, = plt.plot(z, "ro", markersize=3)
# plt.clf()
# plt.close()
#
#
# import random
#
# print(a.shape)
# green_to_red = sns.diverging_palette(145, 10,n=a.shape[0])#, s=70, l=40, n=3
# ax = sns.pairplot(a, hue="d",palette=green_to_red)
# ax._legend.remove()
b = sns.diverging_palette(145, 10,n=10)#, s=70, l=40, n=10
sns.palplot(b[10 - 5:])

sns.plt.show()

# b = pd.DataFrame([[1,'1Frame'],[2,"100Frame"]], columns=['a','d'])
# green_to_red = sns.diverging_palette(145, 10,n=a.shape[0])#, s=70, l=40, n=3
# ax2 = for_legend_data = sns.pairplot(b, hue="d",palette=green_to_red)

# ax.add_legend(title="chinpo", label_order=[1,2]) #legend_data=ax2._legend_data

# sns.plt.legend([green_dot, red_dot, ], ['0', '1'], bbox_to_anchor=(1.3, 1))
# sns.plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# tips = sns.load_dataset("tips")
#
# sns.stripplot(x="day", y="total_bill", hue="smoker",
# data=tips, jitter=True,
# palette="Set2", split=True,linewidth=1,edgecolor='gray')
#
# # Get the ax object to use later.
# ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
# data=tips,palette="Set2",fliersize=0)
#
# # Get the handles and labels. For this example it'll be 2 tuples
# # of length 4 each.
# handles, labels = ax.get_legend_handles_labels()
#
# # When creating the legend, only use the first two elements
# # to effectively remove the last two.
# l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.set(style="whitegrid")
#
# titanic = sns.load_dataset("titanic")
#
# g = sns.factorplot("class", "survived", "sex",
#                    data=titanic, kind="bar",
#                    size=6, palette="muted",
#                    legend=False)
# g.despine(left=True)
# plt.legend(loc='upper left')
# g.set_ylabels("survival probability")
# plt.show()

