import csv

from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt

film_data=open('film.csv','rt')
reader = csv.reader(film_data)

#表头信息
headers = next(reader)
print(headers)

feature_list = [] #特征值
result_list = [] #结果

for row in reader:
    result_list.append(row[-1])

    #去掉首尾两列，特征集中保留‘type','county'，’gross‘
    feature_list.append(dict(zip(headers[1:-1],row[1:-1])))

print(result_list)
print(feature_list)

vec = DictVectorizer() #将dict数据类型的list数据，转换成numpy array
dummyX = vec.fit_transform(feature_list).toarray()
dummyY = preprocessing.LabelBinarizer().fit_transform(result_list)

#注意，dummyX是按首字母排序的 'country','gross','type'

print(dummyX)
print(dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
clf = clf.fit(dummyX,dummyY)
print("clf:"+str(clf))

##开始预测
A=([[0,0,0,1,0,1,0,1,0]]) #日本-低票房-动画片
B=([[0,0,1,0,0,1,0,1,0]]) #法国-低票房-动画片
C=([[1,0,0,0,1,0,1,0,0]]) #美国-高票房-动作片

predict_result=clf.predict(A)
print("预测结果"+str(predict_result))
