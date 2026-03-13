FROM nginx:alpine

# Nginx config with /scan reverse proxy to api-gateway
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Static assets
COPY . /usr/share/nginx/html

EXPOSE 80
