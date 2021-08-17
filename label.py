import os
from collections import Counter
import  pandas as pd
origindata=pd.DataFrame()
index=0
for txtfile in os.listdir("dataset"):
    with open(os.path.join("dataset",txtfile)) as f:
        text_list = f.readlines()
        '''将读取的内容转换为字符串'''
        text_str = ''.join(text_list)
        count = Counter(text_str)
        d_order = sorted(count.items(), key=lambda x: x[0], reverse=False)
        data=pd.DataFrame(d_order).T
        data.drop(labels=None, axis=0, index=0, columns=None, inplace=True)

        index+=1
        origindata=pd.concat([origindata, data],join='outer')

        # print(origindata)
origindata.index=range(index)
for index in range(10,256):
    origindata[str(index)]=0
origindata.to_csv("filelabel.csv")




