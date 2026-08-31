# ── Stage 1: Build ───────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# ── Stage 2: Serve ───────────────────────────────────────────
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template

ENV PORT=80
ENV BACKEND_UPSTREAM=127.0.0.1:8000

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
