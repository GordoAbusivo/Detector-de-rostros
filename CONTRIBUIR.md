# Cómo trabajar con Git en este proyecto

## Reglas
1. **Cada equipo trabaja en su carpeta.** Si necesitan cambiar `comun.py`, avisen en el grupo antes.
2. **No subir fotos ni modelos** (el `.gitignore` ya los bloquea). El dataset se comparte por Drive.
3. **Hagan `git pull` antes de empezar** y `git push` al terminar. Commits chicos y frecuentes.
4. **Una rama por equipo:** `preprocesamiento`, `entrenamiento`, `reporte`. El coordinador junta todo en `main`.
5. Mensajes de commit claros: `preprocesamiento: agrega recorte con margen`, no `cambios`.

## Comandos básicos

```bash
# Primera vez
git clone <URL-del-repo>
cd proyecto-rostros
git checkout -b entrenamiento        # o preprocesamiento / reporte

# Cada día
git pull origin main                 # traer lo nuevo
# ... trabajar ...
git add .
git commit -m "entrenamiento: SVM con HOG listo"
git push origin entrenamiento
```

## Si hay conflicto
No borren cambios de otros. Abran el archivo, busquen `<<<<<<<`, quédense con lo correcto de ambas partes,
guarden, y hagan `git add` + `git commit`. Si dudan, pregunten al coordinador.

## Orden de trabajo entre equipos
`preprocesamiento` → deja `datos/particiones/` listo → `entrenamiento` → deja `resultados/` → `reporte`.
Mientras esperan, cada equipo avanza en lo que no depende del anterior (ver el README de cada carpeta).
