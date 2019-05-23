#ニューラルネットワークのモデルを定義するのに使用
from keras.models import Sequential
#畳み込みの処理・プーリングをする処理するための関数
from keras.layers import Conv2D, MaxPooling2D
#活性化関数、ドロップアウト処理の関数、一次元に変換するための関数、全結合する関数
from keras.layers import Activation, Dropout, Flatten, Dense
#データを扱うためにインポート
from keras.utils import np_utils
import keras
#npと言う名前で参照できるようにする
import numpy as np

classes = ["monkey","boar","crow"]      #検索した単語
num_classes = len(classes)              #クラスの数を変数へ
image_size = 50                         #計算時間短縮のためサイズを小さく変更

#メインの関数を定義
def main():
    #ファイルからデータを配列に読み込む
    X_train, X_test, y_train, y_test = np.load("./animal.npy")
    #256階調の整数値を正規化して０〜１（ニューラルネットワークで計算する場合に誤差が出にくい）
    #astype型に変換することで計算できる
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    #to_categorical:正解値は１、他は０の行列に変換 (one-hot-vector)
    #ターゲットが１の場合[1,0,0]、２の場合[0,1,0]、３の場合[0,0,1]に変換
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    #トレーニングを実行する関数の呼び出し
    model = model_train(X_train, y_train)
    #評価をする関数の呼び出し
    model_eval(model, X_test, y_test)

def model_train(X, y):
    #
    model = Sequential()
    #一層目の定義
    #ニューラルネットワークの層を足すadd
    #第１引数：出力フィルターの数。conv2Dのクラスで32個のフィルターを指定
    #第２引数：フィルタ(カーネル)のサイズ。大きさは3*3、
    #第３引数：paddingで畳み込み結果が同じサイズになるようにピクセルを左右に足すための'same'
    #　→ぜろパディング：計算の回数が増える事により、データの橋の特徴も抽出できる
    #第４引数：input_shapeで入力データの形状を指定
    #　→画像が450枚、行数が50、列数が50、チャンネルが3（RGB）の配列の形状から、「50,50,3」を抜き出す
    # ※ストライドの幅(フィルタを動かすピクセル数)は省略されている
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    #活性化関数：閾値を境にして出力が切り替わる関数
    #　"Relu関数"：正を通して負のところは０というレイヤーを足す
    #(入力が正の時に入力と同じ値を、入力が負の時に０を出力する関数)
    model.add(Activation('relu'))

    # 2層目の定義
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    #プーリングの値の一番大きいものを取り出す（2*2ずつ切り取った中から、値が一番大きいものを取り出して圧縮）
    #情報を圧縮する事で計算コストを削減する。プーリング層はなくても良い
    #Convolution層とPooling層で特徴を検出する働きをする
    model.add(MaxPooling2D(pool_size=(2,2)))
    #Dropoutで２５％を捨ててデータの偏りを減らす
    #一定割合のノードを不活性化させながら学習を行うことで過学習を防ぎ（緩和し）、精度をあげる
    model.add(Dropout(0.25))

    # 64：使用する出力フィルタの数
    # (3,3)：カーネルのサイズ
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #データを一列に並べる　一次元配列にする
    model.add(Flatten())
    #全結合層
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #最後の出力層のノードは３つ（クラスの数）
    model.add(Dense(3))
    #それぞれの画像と一致してる確率を足し混むと１になる
    model.add(Activation('softmax'))

    #トレーニング時の更新アルゴリズム（最適化の手法）
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    #モデルの最適化　loss:損失関数（正解と推定値との誤差）　metrics(正答率)
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=100)

    model.save('./animal_cnn.h5')

    return model

def model_eval(model, X, y):
    #評価の処理　結果をscoresへ。　verbose＝１は途中経過を表示する
    scores = model.evaluate(X, y, verbose=1)
    #損失値
    print('Test Loss: ', scores[0])
    #精度
    print('Test Accuracy: ', scores[1])

#このプログラムが直接呼ばれた時だけmain()実行
if __name__ == "__main__":
    main()
