# --- Etapa 1: Construcción (Build) ---
FROM node:18-alpine AS build
WORKDIR /app

# Copiar dependencias e instalarlas
COPY package*.json ./
RUN npm install

# Copiar el resto del código y compilar
COPY . .
# Usa "npm run build" si usas Vite, o ajusta según tu package.json
RUN npm run build 

# --- Etapa 2: Producción (Nginx) ---
FROM nginx:alpine

# Copiar los archivos compilados desde la etapa anterior
# NOTA: Si usaste Create React App, cambia "/app/dist" por "/app/build"
COPY --from=build /app/dist /usr/share/nginx/html

# Exponer el puerto 80 (el que usa Nginx por defecto)
EXPOSE 80

# Iniciar Nginx
CMD ["nginx", "-g", "daemon off;"]