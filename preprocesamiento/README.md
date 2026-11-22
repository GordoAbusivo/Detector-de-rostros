# Equipo de Preprocesamiento (4 personas)

**Meta:** dejar `datos/particiones/{train,val,test}/<persona>/*.jpg` listo cuanto antes. Los otros dos equipos dependen de esto.

| Integrante | Rol | Nombre |
|---|---|---|
| 1 | Auditoría del dataset | |
| 2 | Detección y recorte de rostros | |
| 3 | Partición train / val / test | |
| 4 | Control de calidad y documentación | |

Todos los comandos se corren desde la **raíz del repo**.

## Persona 1 — Auditoría
```bash
python preprocesamiento/revisar_dataset.py
```
- Cuenta fotos por persona, detecta imágenes ilegibles y duplicados exactos.
- Reporta al grupo quién tiene muy pocas fotos (menos de ~15) para pedirles más.
- Elimina duplicados e ilegibles de `datos/crudo/` (guardando el aviso en el grupo).
- Salida: `resultados/auditoria_dataset.csv`.

## Persona 2 — Detección y recorte
```bash
python preprocesamiento/recortar_rostros.py                    # Haar (rápido)
python preprocesamiento/recortar_rostros.py --detector mediapipe   # si Haar falla mucho
```
- Recorta un cuadrado alrededor del rostro (con margen) y lo redimensiona a 224×224.
- Revisa `resultados/bitacora_recorte.csv`: las fotos `sin_rostro` hay que probarlas con el otro detector o recortarlas a mano.
- Las `varios_rostros` toman el rostro más grande: verifiquen que sea el correcto.

## Persona 3 — Partición
```bash
python preprocesamiento/particionar.py --bloque 3
```
- Divide cada persona en 70 % train / 15 % val / 15 % test.
- `--bloque N` mantiene juntas N fotos consecutivas (por nombre de archivo). Úsenlo si las fotos se tomaron en ráfaga: así no hay fotos casi idénticas en train y test (eso infla el accuracy y el profesor lo nota).
- Semilla fija (42): todos obtienen la misma partición.

## Persona 4 — Control de calidad y documentación
- Abrir ~20 recortes al azar por persona y confirmar que el rostro y la etiqueta son correctos.
- Confirmar que ninguna foto aparece en dos particiones.
- Subir el dataset procesado a Drive y poner el enlace en el grupo.
- Redactar la sección **Dataset y preprocesamiento** del reporte (cuántas fotos, cuántas personas, cómo se recortó, cómo se dividió) y pasársela al equipo de reporte.

## Mientras no haya dataset completo
Trabajen con 2-3 personas en `datos/crudo/` para probar los scripts y avisar a entrenamiento cuando haya una muestra.
