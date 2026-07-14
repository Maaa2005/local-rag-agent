# 項目10: コンテナ最小権限化。非特権ポート (8080) で動作する
# 非rootの nginx イメージを使う (nginx.conf も listen 8080 に合わせて変更済み)。
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY src/ /usr/share/nginx/html/
