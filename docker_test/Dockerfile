#FROM：媒体にするイメージ名。ローカルになければ dockerhub から取ってくる
FROM centos

#MAINTAINER：メンテナンスする人、このファイルを作成した人の名前を記述。
MAINTAINER k-suzuki

#RUN：イメージ起動後に実行するコマンド。今回はイメージ起動時にEPELリポジトリとnginxをyumでインストールします。
RUN yum install -y epel-release
RUN yum install -y nginx

#ADD：ファイルを指定したパスに配置するコマンド。今回はローカルにある先ほど作成したindex.htmlを指定したパスに配置します。
ADD index.html /usr/share/nginx/html/

#ENTRYPOINT：コンテナが実行するファイルを設定します。
ENTRYPOINT ["/usr/sbin/nginx", "-g", "daemon off;", "-c", "/etc/nginx/nginx.conf"]
