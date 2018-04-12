import pandas as pd
import numpy as np
#特征列表
features=['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
"""
accommodates:可以容纳的旅客
bedrooms:卧室的数量
bathrooms:厕所的数量
beds:床的数量
price：每晚的费用
minimum_nights:客人最少租了几天
maximum_nights:客人最多租了几天
number_of_review:评论数量
"""
dc_listings=pd.read_csv('listings.csv')
dc_listings=dc_listings[features]
#3723条数据 每条数据8个特征
print(dc_listings.shape)
dc_listings.head()
#目前我们要求我们的房间容纳三人
our_acc_value=3
dc_listings['distance']=np.abs(dc_listings.accommodates-our_acc_value)
#value_counts()统计数字 sort_index()低到高排序
#例如：和我们距离为0的（同样数量的房间）有461个
# print(dc_listings.distance.value_counts().sort_index())
#sample() 随机洗牌
dc_listings=dc_listings.sample(frac=1,random_state=0)
#根据‘distance’属性进行排序
dc_listings=dc_listings.sort_values('distance')
print(dc_listings)
print(dc_listings.price.head())
# | 择1匹配的管道符号
dc_listings['price']=dc_listings.price.str.replace("\$|,",'').astype(float)
mean_price=dc_listings.price.iloc[:5].mean()
print(mean_price)
#首先制定好训练集 和测试集
# axis=0代表往跨行（down)，而axis=1代表跨列（across)，
dc_listings.drop('distance',axis=1)
#iloc[:2792] 选取2792行之前的作为训练集 之后的作为测试级
train_df=dc_listings.copy().iloc[:2792]
test_df=dc_listings.copy().iloc[2792:]
#基于单变量预测价格
def predict_price(new_listing_value,feature_column):
    temp_df=train_df
    temp_df['distance']=np.abs(dc_listings[feature_column]-new_listing_value)
    temp_df=temp_df.sort_values('distance')
    knn_5=temp_df.price.iloc[:5]
    predict_price=knn_5.mean()
    return predict_price
print(test_df.accommodates[:5])
test_df['predicted_price']=test_df.accommodates.apply(predict_price,feature_column='accommodates')
#预测完毕，接下来进行模型评估
test_df['squared_error']=(test_df['predicted_price']-test_df['price'])**2
mse=test_df['squared_error'].mean()
rmse=mse**(1/2)
print(rmse)
#特征的选择不同，得到的评估结果不同
for feature in ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']:
    test_df['predicted_price']=test_df[feature].apply(predict_price,feature_column=feature)
    test_df['squared_error']=(test_df['predicted_price']-test_df['price'])**2
    mse=test_df['squared_error'].mean()
    rmse=mse**(1/2)
    print('RMSE for the {} column {}'.format(feature,rmse))
# RMSE for the accommodates column 212.9892796705153
# RMSE for the bedrooms column 199.80935328065047
# RMSE for the bathrooms column 230.2471670568422
# RMSE for the beds column 227.7858424753905
# RMSE for the price column 86.11284780095812
# RMSE for the minimum_nights column 252.0041077700427
# RMSE for the maximum_nights column 230.38460161576367
# RMSE for the number_of_reviews column 235.91327066995507
#结果差异比较大，综合利用所有的信息一起来进行测试
#特征的差异值不同 会导致结果差距较大






