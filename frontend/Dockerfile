# =============================================================================
# SentinelOps - Dashboard
# =============================================================================
# Builds the React console and serves the static bundle through nginx.
#
#     docker build -t sentinelops-ui --build-arg VITE_API_BASE_URL=http://localhost:8000 frontend
#
# The API base URL is baked in at BUILD time, not runtime: Vite inlines
# import.meta.env into the bundle. Pointing an existing image at a different
# backend therefore means rebuilding, not restarting.
# =============================================================================

FROM node:22-alpine AS build

WORKDIR /app

# npm ci needs both files and installs exactly what the lockfile pins.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_GITHUB_REPO=prithish47/Mlops-proj
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_GITHUB_REPO=$VITE_GITHUB_REPO

RUN npm run build


FROM nginx:1.27-alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
