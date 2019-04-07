from PIL import Image                   #画像を扱うパッケージ"pillow"
import os, glob                         #ファイルを扱う、ファイルの一覧を取得する
import numpy as np                      #配列を扱うため
#from sklearn import cross_validation
from sklearn import model_selection     #データをトレーニング用とテストに分けるためのツール

classes = ["monkey","boar","crow"]      #検索した単語
num_classes = len(classes)              #クラスの数を変数へ
image_size = 50                         #計算時間短縮のためサイズを小さく変更
num_testdate = 100

#画像の読み込み

X_train = []                                  #画像データを入れる空の配列
X_test = []
Y_train = []                                  #ラベルデータを入れる空の配列（検索した単語の種類がわかる）
Y_test = []

for index, classlabel in enumerate(classes):                 #クラスを順番に取り出して、同時に番号も入れる
                                                             #enumerateを使うとリストのindexが取れる
    phtos_dir = "./" + classlabel                            #画像のディレクトリ
    files = glob.glob(phtos_dir + "/*.jpg")             #ファイルの一覧をまとめて取得
    #一覧の中からファイルを取り出す
    for i, file in enumerate(files):
        if i >= 200: break                              #200回を超えたら処理しない
        image = Image.open(file)                        #ファイルを開く
        image = image.convert("RGB")                    #ファイルを赤青緑に変換
        image = image.resize((image_size, image_size))  #サイズを揃える
        data = np.asarray(image)                        #イメージを数字の配列に変換して入れる

        if i < num_testdate:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20,20,5):
                # 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)



#X = np.array(X)      #TensorFlowが扱いやすいデータ型に揃える
#Y = np.array(Y)      #TensorFlowが扱いやすいデータ型に揃える
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

#サイキットランに含まれるデータを分けるtrain_test_split関数で
#XとYを3：1（トレーニングデータ：正解ラベル）に分割
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)


xy = (X_train, X_test, y_train, y_test)             #4つを一つの関数にまとめる
np.save("./animal_aug.npy", xy)                         #npの配列をファイルに出力
