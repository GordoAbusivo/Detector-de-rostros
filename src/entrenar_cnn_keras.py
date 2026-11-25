"""Modelo D: CNN en TensorFlow / Keras para reconocer a las 29 personas a partir de su CARA recortada.

Lee las caras que deja kaggle_preparar_caras.py (data/caras/ + data/caras_manifest.csv) y entrena con la MISMA regla de
division que el resto del repo: por SETS completos (60/20/20 por persona, ver dividir_sets.py), nunca por foto al azar.

  --modo scratch    CNN propia entrenada desde cero (4 bloques Conv-BN-ReLU + pooling global). Linea base.
  --modo finetune   CNN preentrenada en ImageNet (MobileNetV2, ResNet50V2 o EfficientNetB0): primero se entrena solo la
                    cabeza nueva (--congelar epocas) y luego toda la red con lr bajo. Es la opcion recomendada.

La normalizacion de pixeles va DENTRO del modelo (la red recibe pixeles 0-255) y el aumento de datos tambien (solo se
activa al entrenar), asi que al predecir basta con pasarle la cara recortada, sin preprocesar.

Salida:
    models/<nombre>.keras, models/<nombre>_clases.json     el modelo (BIOMETRICO: no se sube a GitHub)
    resultados/<nombre>/metricas.json, historial.csv, curvas.png, por_clase.csv, matriz_confusion.csv, errores.csv
donde <nombre> = keras_scratch | keras_mobilenetv2 | keras_resnet50v2 | keras_efficientnetb0. Los resultados tienen el
mismo formato que ArcFace y la CNN de PyTorch, asi que `python src/comparar_modelos.py` los junta en una tabla.

Uso (en Kaggle con GPU, ver kaggle/entrenar_cnn_keras_kaggle.ipynb):
    python src/entrenar_cnn_keras.py --modo scratch --tam 128 --epocas 60
    python src/entrenar_cnn_keras.py --modo finetune --arq mobilenetv2 --tam 224 --epocas 20
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import tensorflow as tf                 # noqa: E402
import keras                            # noqa: E402
from keras import layers                # noqa: E402

from comun import DATA, MODELOS, RESULTADOS, guardar_resultados   # noqa: E402
from dividir_sets import asignar_split                            # noqa: E402

# arquitectura -> (constructor de keras.applications, normalizacion que necesita)
#   "tf"   : pixeles 0-255 -> [-1, 1]   (MobileNetV2 y ResNet50V2)
#   "ninguna": EfficientNetB0 ya trae su propia normalizacion y espera 0-255
ARQUITECTURAS = {
    "mobilenetv2": (keras.applications.MobileNetV2, "tf"),
    "resnet50v2": (keras.applications.ResNet50V2, "tf"),
    "efficientnetb0": (keras.applications.EfficientNetB0, "ninguna"),
}


# ------------------------------------------------------------------ datos
def dividir_por_sets(datos):
    """Agrega la columna `split` con la division por sets del repo y garantiza train/val/test para TODAS las personas
    (si alguien queda sin validacion, un set completo de su train, el mas chico, pasa a validacion)."""
    datos = datos.copy()
    datos["split"] = asignar_split(datos)
    for persona in sorted(datos["persona"].unique()):
        de_p = datos["persona"] == persona
        if not (de_p & (datos["split"] == "val")).any():
            tam = datos[de_p & (datos["split"] == "train")].groupby("set").size().sort_values()
            if len(tam) < 2:
                sys.exit(f"ERROR: {persona} tiene menos de 2 sets en train; no se puede sacar validacion.")
            datos.loc[de_p & (datos["set"] == tam.index[0]), "split"] = "val"
            print(f"  {persona}: el set {tam.index[0]} ({tam.iloc[0]} caras) pasa de train a val")
    tabla = pd.crosstab(datos["persona"], datos["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    if (tabla == 0).any().any():
        sys.exit("ERROR: hay personas sin fotos en alguna particion (pocos sets):\n" + tabla[(tabla == 0).any(axis=1)].to_string())
    return datos


def crear_dataset(rutas, etiquetas, n_clases, tam, batch, entrenar, pesos_clase=None, cache=True, semilla=42):
    """tf.data: lee el JPG, lo ajusta a tam x tam (uint8, se guarda en cache) y arma lotes. Solo train se baraja
    (DESPUES del cache, para que el orden cambie en cada epoca) y lleva peso por muestra para compensar clases."""
    ds = tf.data.Dataset.from_tensor_slices((np.array(rutas), np.array(etiquetas, dtype=np.int32)))

    def leer(ruta, y):
        img = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
        img = tf.image.resize(img, (tam, tam), antialias=True)
        return tf.cast(tf.round(img), tf.uint8), y

    ds = ds.map(leer, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if entrenar:
        ds = ds.shuffle(2048, seed=semilla, reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def preparar(x, y):
        x = tf.cast(x, tf.float32)
        if pesos_clase is not None:
            return x, tf.one_hot(y, n_clases), tf.gather(pesos_clase, y)
        return x, tf.one_hot(y, n_clases)

    return ds.map(preparar, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ modelos
def aumento_de_datos():
    """Aumento suave y realista para caras (sin volteo vertical ni giros grandes: nadie sale de cabeza)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="aumento")


def modelo_scratch(tam, n_clases):
    def bloque(x, filtros, drop):
        for _ in range(2):
            x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        return layers.Dropout(drop)(x)

    entrada = keras.Input((tam, tam, 3))
    x = aumento_de_datos()(entrada)             # el aumento trabaja con pixeles 0-255, por eso va ANTES de normalizar
    x = layers.Rescaling(1 / 255.0)(x)
    for filtros, drop in ((32, 0.1), (64, 0.15), (128, 0.2), (256, 0.25)):
        x = bloque(x, filtros, drop)
    x = layers.GlobalAveragePooling2D()(x)      # en vez de Flatten + Dense gigante: ~1.2 M parametros en total
    x = layers.Dropout(0.4)(x)
    return keras.Model(entrada, layers.Dense(n_clases, activation="softmax")(x), name="cnn_scratch")


def modelo_finetune(arq, tam, n_clases, pesos):
    constructor, norm = ARQUITECTURAS[arq]
    base = constructor(include_top=False, weights=None if pesos == "none" else "imagenet", input_shape=(tam, tam, 3))
    base.trainable = False
    entrada = keras.Input((tam, tam, 3))
    x = aumento_de_datos()(entrada)
    if norm == "tf":
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    # Con pesos de ImageNet, training=False deja las capas BatchNorm de la base en modo inferencia siempre (receta estandar
    # de fine-tuning). Sin preentrenar no hay estadisticas que conservar, asi que ahi si se dejan entrenar.
    x = base(x, training=False) if pesos == "imagenet" else base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    modelo = keras.Model(entrada, layers.Dense(n_clases, activation="softmax")(x), name=f"cnn_{arq}")
    return modelo, base


def compilar(modelo, lr):
    modelo.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
                   loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                   metrics=["accuracy"])


class GuardarMejor(keras.callbacks.Callback):
    """Guarda el modelo cada vez que mejora val_loss; sirve igual entre las dos fases del fine-tuning."""

    def __init__(self, ruta):
        super().__init__()
        self.ruta, self.mejor = Path(ruta), np.inf

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.mejor:
            self.mejor = logs["val_loss"]
            self.model.save(self.ruta)


def graficar(historial, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = historial["epoca"]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for eje, cols, titulo in ((ejes[0], ("loss", "val_loss"), "Perdida"), (ejes[1], ("accuracy", "val_accuracy"), "Accuracy")):
        eje.plot(ep, historial[cols[0]], "r", label="train")
        eje.plot(ep, historial[cols[1]], "g", label="validacion")
        mejor = historial[cols[1]].idxmin() if cols[0] == "loss" else historial[cols[1]].idxmax()
        eje.scatter(ep[mejor], historial[cols[1]][mejor], c="b", s=60, zorder=3, label=f"mejor epoca = {ep[mejor]}")
        eje.set_title(titulo)
        eje.set_xlabel("epoca")
        eje.legend()
        eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modo", choices=["finetune", "scratch"], default="finetune")
    ap.add_argument("--arq", choices=list(ARQUITECTURAS), default="mobilenetv2", help="solo en modo finetune")
    ap.add_argument("--caras", default=str(DATA / "caras"), help="carpeta con las caras recortadas")
    ap.add_argument("--manifiesto", default=str(DATA / "caras_manifest.csv"))
    ap.add_argument("--tam", type=int, default=None, help="lado de entrada; por defecto 224 (finetune) o 128 (scratch)")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--congelar", type=int, default=3, help="finetune: epocas iniciales entrenando solo la cabeza")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="por defecto 1e-4 (finetune) o 1e-3 (scratch)")
    ap.add_argument("--paciencia", type=int, default=6, help="epocas sin mejorar val_loss antes de parar")
    ap.add_argument("--pesos", choices=["imagenet", "none"], default="imagenet",
                    help="none = misma arquitectura SIN preentrenar (solo para pruebas rapidas)")
    ap.add_argument("--sin-cache", action="store_true", help="no guardar las imagenes en RAM (si se acaba la memoria)")
    ap.add_argument("--max-por-persona", type=int, default=None, help="solo N caras por persona (pruebas rapidas)")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    finetune = a.modo == "finetune"
    tam = a.tam or (224 if finetune else 128)
    lr = a.lr or (1e-4 if finetune else 1e-3)
    nombre = f"keras_{a.arq}" if finetune else "keras_scratch"
    keras.utils.set_random_seed(a.semilla)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} / Keras {keras.__version__}. GPU: {len(gpus)}" +
          ("" if gpus else "  (sin GPU sera lento; en Kaggle activa Accelerator > GPU)"))

    # ---- datos
    datos = pd.read_csv(a.manifiesto)
    if a.max_por_persona:
        datos = datos.groupby("persona").head(a.max_por_persona)
    datos = dividir_por_sets(datos)
    clases = sorted(datos["persona"].unique())
    indice = {p: i for i, p in enumerate(clases)}
    datos["y"] = datos["persona"].map(indice)
    datos["ruta_abs"] = [str(Path(a.caras) / r) for r in datos["ruta"]]
    partes = {s: datos[datos["split"] == s] for s in ("train", "val", "test")}
    print(f"personas={len(clases)} " + " ".join(f"{s}={len(d)}" for s, d in partes.items()))

    conteo = partes["train"]["y"].value_counts()
    pesos_clase = tf.constant([len(partes["train"]) / (len(clases) * conteo[i]) for i in range(len(clases))], tf.float32)
    kw = dict(n_clases=len(clases), tam=tam, batch=a.batch, cache=not a.sin_cache)
    ds_train = crear_dataset(partes["train"]["ruta_abs"], partes["train"]["y"], entrenar=True,
                             pesos_clase=pesos_clase, semilla=a.semilla, **kw)
    ds_val = crear_dataset(partes["val"]["ruta_abs"], partes["val"]["y"], entrenar=False, **kw)
    ds_test = crear_dataset(partes["test"]["ruta_abs"], partes["test"]["y"], entrenar=False, **kw)

    # ---- modelo
    if finetune:
        modelo, base = modelo_finetune(a.arq, tam, len(clases), a.pesos)
    else:
        modelo, base = modelo_scratch(tam, len(clases)), None
    MODELOS.mkdir(exist_ok=True)
    ruta_modelo = MODELOS / f"{nombre}.keras"
    guardar = GuardarMejor(ruta_modelo)
    historia = []

    def entrenar(epocas_desde, epocas_hasta, callbacks_extra=()):
        h = modelo.fit(ds_train, validation_data=ds_val, epochs=epocas_hasta, initial_epoch=epocas_desde,
                       callbacks=[guardar, *callbacks_extra], shuffle=False, verbose=2)   # el barajado ya va en tf.data
        historia.extend(pd.DataFrame(h.history).assign(epoca=lambda d: range(epocas_desde + 1, epocas_desde + len(d) + 1))
                        .to_dict("records"))

    parar = lambda: keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.paciencia,       # noqa: E731
                                                  restore_best_weights=True)
    bajar_lr = lambda: keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)  # noqa: E731

    fase_cabeza = min(a.congelar, a.epocas) if finetune else 0
    if fase_cabeza:
        print(f"\n== Fase 1: solo la cabeza nueva ({fase_cabeza} epocas, base congelada) ==")
        compilar(modelo, lr * 10)
        entrenar(0, fase_cabeza)
        print("\n== Fase 2: fine-tuning de toda la red ==")
        base.trainable = True
        compilar(modelo, lr)                      # hay que recompilar para que el cambio de trainable surta efecto
    else:
        compilar(modelo, lr)
    if a.epocas > fase_cabeza:
        entrenar(fase_cabeza, a.epocas, [parar(), bajar_lr()])

    # ---- resultados
    carpeta = RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    historial = pd.DataFrame(historia)
    historial.to_csv(carpeta / "historial.csv", index=False)
    graficar(historial, carpeta / "curvas.png")
    (MODELOS / f"{nombre}_clases.json").write_text(json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nMejor val_loss = {guardar.mejor:.4f}. Evaluando en prueba (una sola vez)...")
    mejor = keras.models.load_model(ruta_modelo)
    probas = mejor.predict(ds_test.map(lambda x, y, *_: x), verbose=0)
    predichos = [clases[i] for i in probas.argmax(axis=1)]
    reales = partes["test"]["persona"].tolist()
    guardar_resultados(nombre, reales, predichos, clases, partes["test"]["ruta"].tolist())
    print(f"Modelo guardado en {ruta_modelo} (BIOMETRICO: no lo subas a GitHub ni lo hagas publico).")


if __name__ == "__main__":
    main()
