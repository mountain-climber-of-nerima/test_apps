from sklearn.externals import joblib
import flask
import numpy as np

app = flask.Flask(__name__)
model = None

def load_model():
    '''
    学習済みmodelを読み込む関数
    パスにしていしたpklファイルをmodelに読み込む
    '''
    global model
    print("学習済みmodelを読み込んでいます...")
    model = joblib.load("./model/sample-model.pkl")
    print("ロードが完成しました")

@app.route("/predict", methods=["POST"])
def predict():
    # レスポンスタイプを先に定義
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # POST methodのリクエストを処理
    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            #リクエスト時のjson fileを読み込む
            feature = flask.request.get_json().get("feature")

            # 分類器を入れるときにnumpy arrayに変換する
            feature = np.array(feature).reshape((1, -1))

            #変換したfeatureを用いて、学習済みモデルで推論する
            response["prediction"] = model.predict(feature).tolist()

            # responseのステータスを成功に書き換える
            response["success"] = True

        return flask.jsonnify(response)

if __name__ == "__main__":
    load_model()
    print("サーバーを開始しています.....")
    app.run(host='0.0.0.0', port=5000)

