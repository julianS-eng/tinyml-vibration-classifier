# TinyML Vibration Classifier (resumen en español)

> Este es un resumen en español. La documentación completa y autoritativa
> está en [README.md](README.md) (inglés); los números y figuras de ambos
> documentos provienen de la misma ejecución real del pipeline
> (`docs/results.json`). Para la teoría paso a paso, las decisiones de
> diseño y 10 preguntas de entrevista técnica, ver
> [docs/LEARNING.md](docs/LEARNING.md).

## El problema de negocio

Las paradas no planificadas de maquinaria rotativa (motores, bombas,
ventiladores, reductores) son costosas, y cuando una falla ya se escucha o
un rodamiento se agarrota, la ventana de reparación normalmente ya se
cerró. La vibración es el síntoma observable más temprano: un
**desbalanceo de masa** eleva la vibración a 1x la velocidad del eje mucho
antes de causar daño, y los defectos en rodamientos de elementos rodantes
excitan una resonancia estructural en frecuencias muy específicas y
calculables (BPFI para un defecto en la pista interna, BPFO para la pista
externa) mucho antes de que el rodamiento falle de forma audible. Un
acelerómetro barato más un modelo lo bastante pequeño para correr *en el
mismo nodo sensor* — sin ida y vuelta a la nube, sin transmitir datos crudos
continuamente — convierte esa física en una alerta en tiempo real: normal /
desbalanceo / falla de pista interna / falla de pista externa, corriendo
enteramente en un microcontrolador de unos pocos dólares.

## El pipeline

```
datos (sintéticos, con BPFI/BPFO físicamente correctos -- o CWRU real vía el loader intercambiable)
  |
  +--> features clásicos (RMS, curtosis, factor de cresta, FFT, envolvente de Hilbert)
  |        -> StandardScaler + RandomForestClassifier, CV estratificada 5-fold
  |
  +--> ventana de señal cruda
           -> CNN 1D (Keras) -> TFLite float32 -> TFLite int8 (cuantización con dataset representativo)
                -> arreglo de bytes en C -> firmware/esp32s3 (ESP-IDF + esp-tflite-micro)
```

## Por qué datos sintéticos

El dataset público de rodamientos de Case Western Reserve University (CWRU)
era el objetivo, pero su servidor no era alcanzable desde la red donde se
construyó este proyecto (acceso de salida bloqueado por política). En vez de
omitir los datos en silencio, se implementó un **generador sintético
fundamentado en física**: las frecuencias de falla (BPFI, BPFO) se calculan
con la geometría real del rodamiento SKF 6205-2RS JEM que usa el banco de
pruebas de CWRU, no con números inventados (ver
`tinyml_vibration.data.bearing_physics`). Cada ventana de falla es un tren
de pulsos de resonancia excitados por impacto, repitiéndose exactamente a
BPFI/BPFO — el modelo estándar de señal de falla de rodamientos en la
literatura. Un test automatizado demuestra que el pico del espectro de
envolvente cae a menos de un 5% de la frecuencia BPFI/BPFO teórica.

El loader real de CWRU (`cwru_loader.py`) está implementado contra el
formato `.mat` y la convención de nombres documentados de CWRU, pero por la
misma razón de red **no fue ejecutado contra los archivos reales**: es un
punto de partida documentado, no verificado. Ambos loaders implementan la
misma interfaz `VibrationDataset`, así que cambiar de sintético a real no
requiere tocar el resto del pipeline.

## Resultados reales (misma ejecución que docs/results.json)

**Baseline (Random Forest + features clásicos, CV estratificada 5-fold):**
exactitud 99.83% ± 0.20%, F1 macro 0.9983 ± 0.0020.

**CNN 1D (3,124 parámetros) antes y después de cuantizar a int8:**

| Modelo | Exactitud (test) | Tamaño |
|---|---|---|
| CNN Keras float32 | 100.0% | — |
| CNN TFLite float32 | 100.0% | 18.13 KB |
| CNN TFLite **int8** | 100.0% | **12.36 KB** |

La cuantización no costó exactitud y redujo el tamaño 1.47x (una reducción
menor al ~4x "clásico" porque, a este tamaño de modelo, el overhead fijo del
flatbuffer de TFLite pesa proporcionalmente más). RAM estimada (arena de
activaciones): ~16.0 KB — una *estimación* documentada, no una medición en
hardware real.

**Robustez** (evaluado en un set más difícil: rpm fuera de rango de
entrenamiento y SNR entre -3 y 5 dB): el baseline llega a 91.75% y la CNN
int8 a 88.50% — ambos degradan de forma gradual, no colapsan.

## Firmware ESP32-S3

`firmware/esp32s3/` es un proyecto ESP-IDF completo (con `esp-tflite-micro`)
que embebe los bytes exactos del modelo int8 evaluado arriba, generados
automáticamente desde Python (no escritos a mano), más vectores de prueba
pre-cuantizados por clase para autoverificar la integración del modelo al
arrancar, sin necesitar todavía un sensor físico conectado. CI compila este
proyecto (`idf.py build`) en cada push, en runners de GitHub con acceso de
red al registro de componentes de Espressif. No se flashea ni corre en
hardware real en este repositorio — ver "Future work" en
`firmware/esp32s3/README.md`.

## Limitaciones conocidas

* Los resultados principales son sobre datos **sintéticos**; el path a CWRU
  real está implementado pero no verificado.
* La clase "ball fault" de CWRU se mapea a `imbalance` como la etiqueta más
  cercana disponible — no es el mismo mecanismo físico de falla.
* La RAM es una estimación, no una medición en hardware.
* No hay driver de acelerómetro implementado todavía.
* CI compila el firmware pero no lo flashea ni lo corre.
* 100% de exactitud en el set sintético limpio refleja un problema
  sintético bien separado, no una promesa de 100% en producción — la
  sección de robustez es la señal más representativa.

## Licencia

[MIT](LICENSE)
