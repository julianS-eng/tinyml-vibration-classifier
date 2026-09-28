# Notas de aprendizaje (en español)

Este documento explica, paso a paso, la teoría detrás de este proyecto, las
decisiones de diseño tomadas (y las alternativas que se descartaron y por
qué), y cierra con 10 preguntas de entrevista técnica con sus respuestas.
Está escrito para alguien que entiende de programación pero no
necesariamente de análisis de vibraciones ni de TinyML.

## 1. Teoría paso a paso

### 1.1 ¿Por qué vibración para mantenimiento predictivo?

Toda máquina rotativa vibra. Una máquina sana vibra poco y de forma
predecible (sobre todo a la frecuencia de giro del eje y sus primeros
armónicos). Cuando algo empieza a fallar, la vibración cambia de forma
característica *antes* de que la falla sea audible, visible o catastrófica:

- Un **desbalanceo de masa** (p. ej. un rotor con una zona más pesada que
  otra) genera una fuerza centrífuga que gira con el eje, y por lo tanto una
  vibración senoidal dominante exactamente a la frecuencia de giro (1x).
- Un **defecto en un rodamiento** (una picadura en la pista interna o
  externa) genera un impacto físico cada vez que un elemento rodante (bola o
  rodillo) pasa por el defecto. Ese impacto no se ve como una senoide limpia:
  excita una resonancia estructural de alta frecuencia (típicamente varios
  kHz) que se repite a una frecuencia mucho más baja (cientos de Hz),
  determinada por la geometría del rodamiento y la velocidad del eje.

La idea central de este proyecto es: si esas frecuencias son calculables a
partir de la física del rodamiento, se puede generar un dataset sintético
*correcto* sin necesitar (todavía) el dataset real, y se puede diseñar el
extractor de features para que busque exactamente esas frecuencias.

### 1.2 Cinemática del rodamiento: BPFI, BPFO, BSF, FTF

Para un rodamiento de bolas con la pista externa fija y la interna girando
con el eje a frecuencia `fr` (Hz), con `n` elementos rodantes de diámetro
`d`, diámetro de paso `D` y ángulo de contacto `phi`:

```
BPFO = (n/2) * fr * (1 - (d/D)*cos(phi))   # pista externa
BPFI = (n/2) * fr * (1 + (d/D)*cos(phi))   # pista interna
BSF  = (D/(2d)) * fr * (1 - (d/D)^2*cos(phi)^2)   # giro de la bola
FTF  = (fr/2) * (1 - (d/D)*cos(phi))       # frecuencia de la jaula
```

Una identidad útil para verificar cualquier implementación: `BPFO + BPFI =
n * fr`, siempre, independientemente de la geometría (está en
`tests/test_bearing_physics.py`). Con la geometría del rodamiento SKF
6205-2RS JEM que usa el banco de pruebas de CWRU (9 bolas, `d` = 0.3126 in,
`D` = 1.537 in, `phi` = 0°), a 1797 rpm (29.95 Hz) se obtiene BPFO ≈ 107.3 Hz
y BPFI ≈ 162.2 Hz — los mismos valores que reporta la literatura de
diagnóstico de rodamientos, lo que confirma que la fórmula y la geometría
están bien implementadas.

### 1.3 ¿Por qué el desbalanceo aparece a 1x y no a otra frecuencia?

Una masa desbalanceada en un rotor que gira a `fr` produce una fuerza
centrífuga `F = m * r * omega^2` cuya dirección gira junto con el eje. Visto
desde un sensor fijo, esa fuerza rotante se proyecta como una fuerza
senoidal pura a la frecuencia de giro `fr` (1x) en cualquier eje fijo de
medición. Por eso el desbalanceo es, casi por definición, un problema de "1x
elevado" — y por eso en este proyecto la clase `imbalance` simplemente eleva
la amplitud del armónico 1x (y dos armónicos menores) manteniendo el resto
de la señal como la de una máquina sana.

### 1.4 Features en el dominio del tiempo

- **RMS** (`sqrt(mean(x^2))`): energía global de la vibración. Sube con
  cualquier falla que añada energía, pero no dice *qué tipo* de falla es.
- **Factor de cresta** (`pico / RMS`): para una senoide pura es exactamente
  `sqrt(2) ≈ 1.414` (verificado en `tests/test_classical_features.py`). Sube
  mucho cuando la señal tiene picos aislados de gran amplitud sobre un fondo
  tranquilo — exactamente lo que produce un impacto de rodamiento.
- **Curtosis** (momento de cuarto orden normalizado): mide qué tan
  "puntiaguda" (impulsiva) es una distribución. Una senoide pura tiene
  curtosis de Pearson 1.5; ruido gaussiano blanco, ~3.0; una señal con
  impactos esporádicos de gran amplitud, mucho más que eso (>5, a veces
  >20). Es uno de los indicadores más clásicos y robustos de falla de
  rodamiento precisamente porque no depende de conocer la frecuencia de
  falla de antemano.
- **Factor de forma, factor de impulso, factor de holgura**: variantes
  normalizadas del mismo concepto (relación entre un estadístico de pico y
  uno de tendencia central), cada una con distinta sensibilidad al ruido de
  fondo vs. a los picos.

### 1.5 Dominio de la frecuencia: FFT, resolución y ventaneo

La FFT de una ventana de `N` muestras a frecuencia de muestreo `fs` da
`N/2` bins útiles (por simetría de una señal real), cada uno cubriendo
`fs/N` Hz de resolución. Con `fs = 12000` Hz y `N = 2048`, la resolución es
`12000/2048 ≈ 5.86` Hz — suficiente para separar BPFI (~162 Hz) de BPFO
(~107 Hz) con margen, pero un factor real a la hora de elegir `N`: ventanas
más largas dan más resolución en frecuencia pero cuestan más memoria y más
latencia (hay que esperar a acumular más muestras antes de poder clasificar).
Antes de la FFT se aplica una ventana de Hanning para reducir el *leakage*
espectral (energía de un pico que "se derrama" a bins vecinos por el corte
abrupto de una ventana rectangular) — un detalle estándar de procesamiento
de señales que, si se omite, hace que los picos de BPFI/BPFO se vean más
anchos y más bajos de lo que son.

### 1.6 Análisis de envolvente (demodulación de Hilbert)

Este es el paso menos intuitivo y el más importante para diagnóstico de
rodamientos. Un impacto de rodamiento no aparece en el espectro como un pico
limpio a BPFI/BPFO: aparece como una *resonancia estructural* de alta
frecuencia (varios kHz) cuya *amplitud* está modulada a BPFI/BPFO. Es
exactamente el mismo concepto que la modulación de amplitud (AM) en radio:
hay una portadora de alta frecuencia (la resonancia) y una señal moduladora
de baja frecuencia (la tasa de impactos) montada sobre su envolvente. Mirar
el espectro de la señal cruda muestra la portadora (poco informativa por sí
sola); mirar el espectro de la *envolvente* (magnitud de la transformada de
Hilbert, que es la señal analítica) recupera la moduladora — es decir,
recupera BPFI/BPFO limpiamente. El código en
`tinyml_vibration/features/classical.py::_envelope_features` implementa
exactamente esto, y la figura `docs/img/example_signals.png` lo muestra
visualmente: el pico del espectro de envolvente cae justo sobre la línea
roja punteada (la frecuencia BPFI/BPFO teórica).

### 1.7 CNN 1D sobre la señal cruda vs. features clásicos

El baseline necesita que un humano decida de antemano qué calcular (RMS,
curtosis, bandas de frecuencia...). La CNN 1D, en cambio, aprende sus
propios filtros directamente sobre la forma de onda cruda — en la práctica,
las primeras capas convolucionales terminan aprendiendo algo parecido a
filtros pasa-banda alrededor de las frecuencias relevantes, sin que nadie se
lo diga explícitamente. La ventaja es que generaliza mejor si aparecen
patrones que un ingeniero no anticipó; la desventaja es que es una caja
más negra y, en este proyecto, en el set de robustez generalizó *peor* que
el baseline clásico (88.5% vs 91.75%) — un resultado real, no uno buscado a
propósito, y una buena ilustración de que "más profundo" no es
automáticamente "mejor" cuando los features clásicos ya capturan la física
del problema.

### 1.8 Cuantización int8

Un modelo entrenado en float32 usa 4 bytes por peso. La cuantización int8
"full integer" convierte pesos y activaciones a enteros de 8 bits mediante
una transformación afín: `valor_real ≈ escala * (valor_int8 - zero_point)`.
La `escala` y el `zero_point` se calculan por tensor a partir de un paso de
*calibración*: se corren unas ~200 muestras representativas del set de
entrenamiento por el modelo y se observa el rango real de cada activación.
Esto reduce el tamaño en memoria (menos bytes por peso) y, en la mayoría de
microcontroladores, también la latencia (los kernels enteros están más
optimizados en hardware sin FPU de doble precisión, y hasta con FPU, las
operaciones enteras suelen tener kernels SIMD más eficientes). El costo es
precisión numérica: por eso se mide explícitamente la exactitud antes y
después de cuantizar en vez de asumir que "no debería cambiar mucho".

### 1.9 TensorFlow Lite Micro y el "tensor arena"

TFLite Micro está diseñado para correr sin sistema operativo, sin
`malloc`/`free` dinámicos y sin sistema de archivos. Toda la memoria que el
intérprete necesita para activaciones intermedias durante la inferencia se
reserva de una sola vez, de forma estática, en un buffer llamado el *tensor
arena* (`main/inference.cc`, `g_tensor_arena`). Si el arena es muy pequeño,
`AllocateTensors()` falla en el arranque con un error claro; si es
demasiado grande, se desperdicia RAM que en un microcontrolador es un
recurso escaso (cientos de KB, no GB). Por eso este proyecto reporta una
*estimación* del tamaño de arena necesario (ver
`tinyml_vibration.models.quantize.estimate_peak_activation_bytes`) y además
deja que el firmware imprima el uso *real* medido en el propio dispositivo
al arrancar — la estimación es para planear, la medición en hardware es la
que manda.

## 2. Decisiones de diseño y alternativas descartadas

- **Datos sintéticos en vez de esperar acceso a CWRU.** Alternativa
  descartada: bloquear todo el proyecto hasta conseguir acceso de red al
  servidor de CWRU. Se prefirió construir un generador sintético
  *verificablemente* correcto (con un test que confirma que las frecuencias
  de falla caen donde deben) y dejar el loader real implementado y listo
  para conectar, documentando honestamente que no fue probado contra datos
  reales, en vez de detener el proyecto o simular datos sin ninguna base
  física.
- **Ventana de 2048 muestras a 12 kHz (~171 ms).** Alternativa descartada:
  ventanas más cortas (más rápido, menos memoria, pero peor resolución en
  frecuencia — podría no separar BPFI de BPFO a bajas velocidades de eje) o
  más largas (mejor resolución, pero más RAM y más latencia antes de poder
  clasificar). 2048 fue elegido porque da ~5.9 Hz de resolución, suficiente
  margen frente a la separación BPFI-BPFO (~55 Hz en el caso de referencia),
  sin ventanas de segundos completos.
- **Random Forest como baseline en vez de SVM o gradient boosting.**
  Alternativa descartada: SVM (más sensible a la escala de features aunque
  ya se usa un `StandardScaler`, y menos informativo para interpretabilidad)
  o XGBoost (mejor rendimiento marginal probable, pero una dependencia
  adicional pesada para lo que es, deliberadamente, el modelo "de
  referencia rápida de interpretar", no el modelo final). Random Forest da
  `feature_importances_` gratis, lo cual es valioso para entender *por qué*
  el modelo funciona (ver `docs/img/feature_importance.png`).
- **CNN 1D en vez de LSTM/GRU (recurrente).** Alternativa descartada:
  redes recurrentes, que en teoría capturan dependencias temporales largas
  mejor. Se descartaron porque (a) TFLite Micro tiene soporte limitado y
  más frágil para operadores recurrentes cuantizados a int8 que para
  convolucionales, y (b) una CNN con capas *strided* es más barata
  computacionalmente en una MCU (paralelizable, sin estado secuencial que
  mantener entre pasos de tiempo).
- **Global Average Pooling en vez de Flatten + Dense grande.** Alternativa
  descartada: aplanar el mapa de activaciones de la última capa
  convolucional (256 pasos x 32 canales = 8192 valores) y conectarlo
  directamente a una capa densa. Eso habría añadido decenas de miles de
  parámetros solo en esa capa — dominando por completo el presupuesto de
  memoria del modelo. Global Average Pooling colapsa cada canal a un solo
  valor (promedio en el tiempo) antes de la capa densa, manteniendo el
  modelo en ~3,100 parámetros totales.
- **Cuantización "full integer" (pesos + activaciones + entrada/salida en
  int8) en vez de "dynamic range" o float16.** Alternativa descartada:
  cuantización dinámica (solo pesos en int8, activaciones en float en
  tiempo de inferencia) es más simple pero no reduce tanto el uso de RAM en
  tiempo de ejecución ni acelera tanto en hardware sin soporte float
  eficiente. Full integer es el formato que TFLite Micro espera de forma
  más nativa y predecible para un despliegue en MCU.
- **Estimación de RAM por análisis propio en vez de asumir un número de la
  literatura.** Alternativa descartada: citar una cifra genérica de "TFLite
  Micro típicamente usa X KB". Se prefirió implementar (y documentar
  explícitamente como estimación, no medición) un análisis propio a partir
  del grafo real del modelo generado en esta sesión — trazable y
  reproducible, aunque con limitaciones documentadas.
- **No implementar el driver de acelerómetro.** Alcance descartado
  deliberadamente, no una omisión accidental: la elección de sensor (SPI vs
  I2C, ancho de banda necesario para 12 kHz) depende de una decisión de
  hardware que este repositorio no toma por el usuario. Se documenta
  explícitamente como trabajo futuro en `firmware/esp32s3/README.md` en vez
  de dejarlo como un cabo suelto silencioso.
- **Mapear la clase "ball fault" de CWRU a `imbalance`.** Alternativa
  descartada: agregar una quinta clase específica para falla de bola. Se
  optó por un mapeo explícito y documentado como limitación conocida
  (`README.md`, sección "Known limitations") antes que ignorar esos
  archivos silenciosamente o inventar una clase sin datos sintéticos
  correspondientes que la respalden.

## 3. Preguntas de entrevista técnica (con respuestas)

**1. ¿Qué son BPFI y BPFO, y por qué son distintos entre sí para el mismo
rodamiento?**

BPFI (Ball Pass Frequency, Inner race) y BPFO (... Outer race) son las
frecuencias a las que un elemento rodante pasa por un defecto en la pista
interna o externa, respectivamente. Son distintas porque, cuando la pista
externa está fija y la interna gira con el eje, la velocidad relativa entre
un elemento rodante y cada pista es distinta (la pista interna se mueve
junto con el eje respecto a la jaula, la externa no). Matemáticamente,
`BPFO = (n/2)*fr*(1 - (d/D)cos(phi))` y `BPFI = (n/2)*fr*(1 + (d/D)cos(phi))`
— y siempre se cumple `BPFO + BPFI = n * fr`.

**2. ¿Por qué el desbalanceo se manifiesta específicamente a 1x la
velocidad del eje?**

Porque una masa desbalanceada genera una fuerza centrífuga cuya dirección
gira físicamente junto con el eje a `fr`. Vista desde un sensor fijo, esa
fuerza rotante se proyecta como una componente senoidal pura exactamente a
`fr` (más eventualmente un segundo armónico menor por efectos no lineales
menores) — no aparece "difuminada" en un rango de frecuencias porque el
mecanismo físico que la genera es, en sí mismo, periódico a exactamente esa
frecuencia.

**3. ¿Por qué hace falta demodulación (análisis de envolvente) para
detectar fallas de rodamiento, en vez de mirar directamente el espectro de
la señal cruda?**

Porque el impacto de un defecto de rodamiento no genera una vibración
senoidal limpia a BPFI/BPFO: excita una resonancia estructural de alta
frecuencia (la "portadora"), y es la *amplitud* de esa resonancia la que se
repite a BPFI/BPFO (la "moduladora"). En el espectro de la señal cruda,
BPFI/BPFO no aparece como un pico limpio — aparece enterrado como bandas
laterales alrededor de la resonancia. Al tomar la magnitud de la
transformada de Hilbert (la envolvente) y calcular su espectro, se
recupera directamente la moduladora, es decir, BPFI/BPFO, sin el ruido de
la portadora de por medio.

**4. ¿Qué mide la curtosis y por qué es un buen indicador de falla de
rodamiento incipiente?**

La curtosis mide qué tan concentrada está la "masa" de una distribución en
sus colas frente a su centro — en términos de señal, qué tan impulsiva es.
Una señal con muchos valores pequeños y pocos picos grandes y aislados
(como un impacto de rodamiento entre eventos de rotación normal) tiene
curtosis alta, mientras que una senoide o ruido gaussiano tienen curtosis
baja o moderada. Es especialmente útil para detección *temprana* porque no
necesita conocer de antemano la frecuencia de falla — sube incluso antes de
que el defecto sea lo bastante grande como para producir un pico espectral
claro.

**5. ¿Qué diferencia hay entre cuantización "full integer" y "dynamic
range", y por qué se usó full integer en este proyecto?**

En "dynamic range", solo los pesos se cuantizan a int8; las activaciones se
calculan en float32 en tiempo de inferencia (con una conversión rápida
de/hacia int8 alrededor de cada operación). En "full integer", tanto pesos
como activaciones —y opcionalmente entrada y salida— quedan en int8 de
forma persistente, calibrados con un dataset representativo. Se usó full
integer porque TFLite Micro (el runtime en el microcontrolador) espera
este formato de forma más nativa y predecible, y porque reduce más el uso
de RAM en tiempo de ejecución, que es el recurso más escaso en un MCU.

**6. ¿Cómo afecta la longitud de la ventana de análisis (número de
muestras) al sistema, y qué trade-off implica?**

Más muestras por ventana dan mejor resolución en frecuencia (`fs/N` Hz por
bin) — útil para separar frecuencias de falla cercanas entre sí a bajas
velocidades de eje — pero cuestan más memoria para almacenar la ventana, más
cómputo por inferencia, y sobre todo más *latencia de detección*: hay que
esperar a que se acumulen todas las muestras antes de poder clasificar, así
que una ventana más larga retrasa la primera alerta posible.

**7. ¿Por qué se usó Global Average Pooling en vez de aplanar (Flatten) el
mapa de activaciones antes de las capas densas?**

Porque aplanar preserva la dimensión temporal completa (p. ej. 256 pasos x
32 canales = 8192 valores) y conectarla a una capa densa multiplicaría el
número de parámetros por ese factor — dominando el tamaño total del modelo
justo cuando el objetivo es mantenerlo pequeño para un microcontrolador.
Global Average Pooling colapsa cada canal a un único valor (su promedio en
el tiempo), lo que además da cierta invariancia a *dónde* en la ventana
ocurre el patrón relevante, no solo a cuántos parámetros tiene el modelo.

**8. ¿Qué es el "tensor arena" en TensorFlow Lite Micro y por qué se
reserva de forma estática?**

Es el bloque de memoria (un simple arreglo de bytes) donde el intérprete de
TFLite Micro coloca todas las activaciones intermedias necesarias durante
una inferencia. Se reserva de forma estática (en tiempo de compilación,
como un arreglo global) porque TFLite Micro está diseñado para correr sin
un sistema operativo con gestión dinámica de memoria confiable: en un
microcontrolador no hay garantía de que `malloc` esté disponible, y aunque
lo esté, la fragmentación de heap es un riesgo real en un dispositivo que
debe correr indefinidamente sin reiniciarse.

**9. ¿Cómo se puede validar que un generador de datos sintéticos es
"físicamente correcto" y no simplemente ruido con una etiqueta pegada
encima?**

Verificando que las propiedades que la física predice realmente aparecen en
los datos generados, con una métrica cuantitativa, no solo visualmente. En
este proyecto: (a) las fórmulas de BPFI/BPFO se contrastan contra valores
de referencia publicados para la misma geometría de rodamiento (CWRU,
1797 rpm), y (b) se demodula cada señal de falla generada y se confirma
que el pico del espectro de envolvente cae dentro de un margen pequeño
(<5%) de la frecuencia BPFI/BPFO teórica calculada a partir de la
velocidad de eje real de esa ventana — ambas cosas están automatizadas como
tests (`tests/test_bearing_physics.py`, `tests/test_synthetic_data.py`), no
son una inspección manual única.

**10. Más allá de tener un modelo entrenado y cuantizado, ¿qué le falta a
este proyecto para ser un sistema de mantenimiento predictivo en
producción?**

Al menos: (a) datos reales del parque de máquinas objetivo, no solo un
banco de pruebas de laboratorio; (b) un driver de sensor real y validación
end-to-end en hardware (latencia y RAM medidas, no estimadas); (c) lógica de
alertamiento con histéresis/debounce a nivel de flota, para que una ventana
ruidosa aislada no dispare una alerta — normalmente exigiendo varias
clasificaciones consecutivas o un umbral de confianza antes de notificar;
(d) un mecanismo de actualización de modelo (OTA) y versionado, porque el
modelo actual queda horneado en el firmware; y (e) un ciclo de
reentrenamiento con datos de campo, porque ningún dataset de laboratorio
cubre todos los modos de operación reales de una flota en producción.
