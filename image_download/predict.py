#ニューラルネットワークのモデルを定義するのに使用
from keras.models import Sequential,load_model
#畳み込みの処理・プーリングをする処理するための関数
from keras.layers import Conv2D, MaxPooling2D
#活性化関数、ドロップアウト処理の関数、一次元に変換するための関数、全結合する関数
from keras.layers import Activation, Dropout, Flatten, Dense
#データを扱うためにインポート
from keras.utils import np_utils
import keras
#npと言う名前で参照できるようにする
import numpy as np
#画像を扱うパッケージ"pillow"
from PIL import Image
import sys

classes = ["monkey","boar","crow"]      #検索した単語
num_classes = len(classes)              #クラスの数を変数へ
image_size = 50                         #計算時間短縮のためサイズを小さく変更


def build_model():
    model = Sequential()
    #ニューラルネットワークの層を足すadd
    #conv2Dのクラスで32個のフィルターの3*3、
    #paddingで畳み込み結果が同じサイズになるようにピクセルを左右に足す
    #input_shapeで入力データの形状を指定
    model.add(Conv2D(32,(3,3), padding='same',input_shape=(50,50,3)))
    #活性化関数　正を通して負のところは０というレイヤーを足す
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    #プーリングの値の一番大きいものを取り出す
    model.add(MaxPooling2D(pool_size=(2,2)))
    #Dropoutで２５％を捨ててデータの偏りを減らす
    model.add(Dropout(0.25))


    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #データを一列に並べる
    model.add(Flatten())
    #全結合層
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #最後の出力層のノードは３つ
    model.add(Dense(3))
    #それぞれの画像と一致してる確率を足し混むと１になる
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    #モデルの最適化　loss:損失関数（正解と推定値との誤差）　metrics(正答率)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    # モデルのロード
    model = load_model('./animal_cnn_aug.h5')

    return model

def main():
    # コマンドラインの引数２番目のファイルを開く
    image = Image.open(sys.argv[1])
    # RGBに変換
    image = image.convert('RGB')
    # 画像のサイズを変換
    image = image.resize((image_size, image_size))
    data = np.asarray(image)/255
    X = []
    # Xのリストに追加
    X.append(data)
    # Xをnpの配列に変換
    X = np.array(X)
    # build_modelを呼ぶ
    model = build_model()
    # 推定結果を格納する
    result = model.predict([X])[0]
    # 一番値の大きい配列（推定確率の高いもの）の添字を返す
    predicted = result.argmax()
    # 確率とラベル名を表示
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()
