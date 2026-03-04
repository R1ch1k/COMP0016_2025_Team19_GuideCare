FROM node:20-alpine

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package.json package-lock.json* ./
RUN npm install

# Copy source
COPY . .

# Build-time env vars (Next.js bakes NEXT_PUBLIC_ vars at build time)
ARG NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
