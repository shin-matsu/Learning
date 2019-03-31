from flickrapi import FlickrAPI              #FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint                   #途中のデータを表示させるため
import os, time, sys                        #OSの情報を得る

#APIキー
key = "1f9ab2e4c7696298e6bc001030c1e1e0"    #FlickrAPIキー
secret = "5d0d3dfd58dd68d4"                 #FlickrAPIシークレットキー
wait_time = 1                               #不正アクセスと見なされないた目の間隔

#保存フォルダの指定
imagename = sys.argv[1]                     #２番目の引数から名前をとる
savedir = "./" + imagename

#FlickerのAPIにアクセスするオブジェクト変数
frickr = FlickrAPI(key, secret, format='parsed-json')   #帰ってきたデータをjsonで受け取る

#検索時のパラメータ
result = frickr.photos.search(
    text = imagename,                       #検索する文字列
    per_page = 400,                         #検索件数（何件取得するか）
    media = 'photos',                       #検索するデータの種類
    sort = 'relevance',                     #データの並び順　relevance = 関連する順
    safe_search = 1,                        #有害コンテンツ表示オプション
    extras = 'url_q, licence'               #返り値（画像のアドレス、ライセンス情報）
)

#返り値を表示
photos = result['photos']
#pprint(photos)

#フォルダに画像を追加する
for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']                              #url_qに各画像のアドレスを入れる
    filepath = savedir + '/' + photo['id'] + '.jpg'     #idをファイル名として、jpgで保存
    #ファイルの重複チェック
    if os.path.exists(filepath): continue               #もし重複していればcontinueで次のループへ
    urlretrieve(url_q,filepath)                         #重複してなければurl_qの画像をfilepathの場所へ保存
    time.sleep(wait_time)
