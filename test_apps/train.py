from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

def main():
    # SVMを分類器にする
    clf = svm.SVC()
    # iris datasetを読み込む
    iris = datasets.load_iris()
    # X:従属変数 y:説明変数
    X, y = iris.data, iris.target
    #学習
    clf.fit(X, y)
    # modelをpickleファイルに保存する
    joblib.dump(clf, './model/sample-model.pkl')

if __name__ == '__main__':
    main()