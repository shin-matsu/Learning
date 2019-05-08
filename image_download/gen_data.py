from PIL import Image                   #
import os, glob                         #ファイルを扱う、ファイルの一覧を取得する
import numpy as np                      #配列を扱うため
#from sklearn import cross_validation
from sklearn import model_selection     #データをトレーニング用とテストに分けるためのツール

classes = ["monkey","boar","crow"]      #検索した単語
num_classes = len(classes)              #クラスの数を変数へ
image_size = 50                         #計算時間短縮のためサイズを小さく変更

#画像の読み込み

X = []                                  #画像データを入れる空の配列
Y = []                                  #ラベルデータを入れる空の配列

for index, classlabel in enumerate(classes):                 #クラスを順番に取り出して、同時に番号も入れる
    phtos_dir = "./" + classlabel                            #画像のディレクトリ
    files = glob.glob(phtos_dir + "/*.jpg")             #ファイルの一覧をまとめて取得
    #一覧の中からファイルを取り出す
    for i, file in enumerate(files):
        if i >= 200: break                              #200回を超えたら処理しない
        image = Image.open(file)                        #ファイルを開く
        image = image.convert("RGB")                    #ファイルを赤青緑に変換
        image = image.resize((image_size, image_size))  #画像のサイズを揃える
        data = np.asarray(image)                        #イメージを数字の配列に変換して入れる
        X.append(data)                                  #Xの最後尾に追加
        Y.append(index)                                 #Yの最後尾に追加
                                                        #Yはラベル。順番に数字が入る。猿には０、猪には１、烏には２

X = np.array(X)      #TensorFlowが扱いやすいデータ型に揃える
Y = np.array(Y)      #TensorFlowが扱いやすいデータ型に揃える

#サイキットランに含まれるデータを分けるtrain_test_split関数で
#XとYを3：1（トレーニングデータ：正解ラベル）に分割
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)


xy = (X_train, X_test, y_train, y_test)             #4つを一つの関数にまとめる
np.save("./animal.npy", xy)                         #npの配列をファイルに出力
