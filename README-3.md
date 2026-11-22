# Equipo de Preprocesamiento (4 personas)

**Meta:** dejar `datos/particiones/{train,val,test}/<persona>/*.jpg` listo cuanto antes. Los otros dos equipos dependen de esto.

| Integrante | Rol | Nombre |
|---|---|---|
| 1 | Auditoría del dataset | |
| 2 | Detección y recorte de rostros | |
| 3 | Partición train / val / test | |
| 4 | Control de calidad y documentación | |

Todos los comandos se corren desde la **raíz del repo**.

## Paso 0 — Bajar las fotos
Las fotos están en el dataset privado de Hugging Face (repo de datos: `Dataset-Identificador-Rostros-ESCOM`).
Bájenlas con su `descargar_datos.py` (o `hf download`) y déjenlas en `datos/crudo/` con esta estructura:

```
datos/crudo/<apellido_nombre>/set_01/001.jpg
                            /set_02/001.jpg ...
```
Si ya las tienen en otra carpeta, usen `--entrada <ruta>` en los scripts en vez de copiarlas.
Son fotos de cuerpo completo (~768×1536) y ~1000 por persona: el recorte tarda, déjenlo corriendo.

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
python preprocesamiento/recortar_rostros.py                        # auto: MediaPipe si está instalado, si no Haar
python preprocesamiento/recortar_rostros.py --detector mediapipe   # recomendado para cuerpo completo y disfraces
python preprocesamiento/recortar_rostros.py --detector haar        # si MediaPipe no instala
```
- Recorta un cuadrado alrededor del rostro (con margen) y lo redimensiona a 224×224, conservando la carpeta de cada set.
- Revisa `resultados/bitacora_recorte.csv`: las fotos `sin_rostro` hay que probarlas con el otro detector o recortarlas a mano.
- Las `varios_rostros` toman el rostro más grande: verifiquen que sea el correcto.

## Persona 3 — Partición
```bash
python preprocesamiento/particionar.py
```
- Divide cada persona en ~70 % train / 15 % val / 15 % test **repartiendo sets completos** (`set_01`, `set_02`…). Las fotos de un set son ráfagas casi idénticas; si se mezclaran entre train y test, el accuracy saldría inflado.
- Como cada set puede tener otro disfraz o accesorio, test contiene looks que el modelo no vio: es una evaluación más honesta (y más difícil). Mencionen esto en el reporte.
- Si no hay subcarpetas de sets, reparte fotos sueltas: `--modo foto --bloque 3` mantiene juntas 3 fotos consecutivas.
- Semilla fija (42): todos obtienen la misma partición.

## Persona 4 — Control de calidad y documentación
- Abrir ~20 recortes al azar por persona y confirmar que el rostro y la etiqueta son correctos.
- Confirmar que ninguna foto aparece en dos particiones.
- Subir el dataset procesado a Drive y poner el enlace en el grupo.
- Redactar la sección **Dataset y preprocesamiento** del reporte (cuántas fotos, cuántas personas, cómo se recortó, cómo se dividió) y pasársela al equipo de reporte.

## Mientras no haya dataset completo
Trabajen con 2-3 personas en `datos/crudo/` para probar los scripts y avisar a entrenamiento cuando haya una muestra.
