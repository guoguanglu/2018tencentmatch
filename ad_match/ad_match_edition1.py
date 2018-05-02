# coding=utf-8
# @author:guoguanglu
# github: https://github.com/guoguanglu/2018tencentmatch
#https://blog.csdn.net/bryan__/article/details/79623239
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
########################################读取数据#################################
#############读取用户特征数据############
if os.path.exists('D:/Documents/tengxungame/preliminary_contest_data/userFeature.cvs'):
    userFeature = pd.read_csv('D:/Documents/tengxungame/preliminary_contest_data/userFeature.cvs')
else:
    #userFeature_data为list中的每一个元素为dict，key为name（feature or group feature），value为 feature value
    userFeature_data = []
    with open('D:/Documents/tengxungame/preliminary_contest_data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        #将user_feature转换为行标志为个数，列名为特征组名的DataFrame文件
        user_feature = pd.DataFrame(userFeature_data)
        #将user_feature 转换成csv格式
        user_feature.to_csv('data/userFeature.csv', index=False)
        del userFeature_data
############读取其他数据数据#############
ad_feature=pd.read_csv('D:/Documents/tengxungame/preliminary_contest_data/adFeature.csv')
train = pd.read_csv('D:/Documents/tengxungame/preliminary_contest_data/train.csv')
predict = pd.read_csv('D:/Documents/tengxungame/preliminary_contest_data/test1.csv')
#print ad_feature
##################################################数据处理#######################
############将负标签用0代替，将predict的标签用-1补充
#训练数据中负样本的标签给的是-1，需要先转成0，预测数据的标签置为-1，方便合并后区分数据集。
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
#为每一个数据右面并上用户特征和广告特征，
#然后将广告（adfeature，userfeature）看做一个数据的所有特征，然后进行算法
data=pd.merge(data, ad_feature,on='aid',how='left')
data=pd.merge(data, user_feature,on='uid',how='left')
##将缺失值填充为 '-1' ，为什么不是数值的-1呢？因为在LabelEncoder的时候需要对数据排序
data=data.fillna('-1')
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
#取‘label’列中不为-1的行,即取出train数据，此时已经得到了新的合并特征数据集
train=data[data.label!=-1]
train_y=train.pop('label')#为train的label,结构跟list差不多用法
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)#删除test中之间定义的label