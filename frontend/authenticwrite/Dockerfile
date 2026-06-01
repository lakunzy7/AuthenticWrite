# Multi-stage: build React app, then serve with nginx

# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./

RUN npm ci

COPY public ./public
COPY src ./src
COPY tsconfig.json tailwind.config.js postcss.config.js ./

RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config: serve / (SPA), proxy /api/* to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built React app from builder
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
