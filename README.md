# Transformer from Scratch (PyTorch)

Este repositorio contiene una implementacion de un Transformer tipo Encoder-Decoder en PyTorch, definida en [modelo.py](modelo.py).

## Contenido

- [modelo.py](modelo.py): implementacion completa de bloques del Transformer
- [Attentionisallyouneed.pdf](Attentionisallyouneed.pdf): paper de referencia

## Que implementa

El archivo [modelo.py](modelo.py) incluye estos componentes:

1. `LayerNormalization`
2. `FeedForwardBlock`
3. `InputEmbeddings`
4. `PositionalEncoding` (sinusoidal)
5. `ResidualConnection`
6. `MultiHeadAttentionBlock`
7. `EncoderBlock` y `Encoder`
8. `DecoderBlock` y `Decoder`
9. `ProjectionLayer`
10. `Transformer`
11. `build_transformer(...)` para construir el modelo completo

## Explicacion de cada clase

### 1) `LayerNormalization`

Normaliza cada vector de caracteristicas por token en la ultima dimension.

- Objetivo: estabilizar el entrenamiento evitando activaciones con escala inestable.
- Parametros clave:
    - `features`: tamano de la ultima dimension (normalmente `d_model`).
    - `eps`: valor pequeno para evitar division por cero.
- Parametros aprendibles:
    - `alpha`: escala por caracteristica.
    - `bias`: desplazamiento por caracteristica.
- Entrada tipica: `(batch, seq_len, hidden_size)`.
- Salida: misma forma que la entrada.

### 2) `FeedForwardBlock`

Bloque MLP por posicion aplicado de forma independiente a cada token.

- Flujo interno:
    - `Linear(d_model -> d_ff)`
    - `ReLU`
    - `Dropout`
    - `Linear(d_ff -> d_model)`
- Funcion: aumentar capacidad no lineal entre capas de atencion.
- Entrada: `(batch, seq_len, d_model)`.
- Salida: `(batch, seq_len, d_model)`.

### 3) `InputEmbeddings`

Convierte ids de tokens en vectores densos.

- Usa `nn.Embedding(vocab_size, d_model)`.
- Multiplica por `sqrt(d_model)` para escalar embeddings, siguiendo el paper.
- Entrada: `(batch, seq_len)` con ids enteros.
- Salida: `(batch, seq_len, d_model)`.

### 4) `PositionalEncoding`

Inyecta informacion de orden en la secuencia con funciones sinusoidales.

- Genera una matriz fija `pe` de forma `(1, seq_len, d_model)`.
- Se registra como buffer (`register_buffer`) para que:
    - se guarde con el modelo,
    - no sea un parametro entrenable.
- En `forward`, suma `pe` a los embeddings y aplica dropout.
- Entrada: `(batch, seq_len, d_model)`.
- Salida: `(batch, seq_len, d_model)`.

### 5) `ResidualConnection`

Aplica patron residual con normalizacion y dropout.

- Formula usada: `x + dropout(sublayer(norm(x)))`.
- Funcion: facilitar el flujo de gradiente y estabilizar profundidad.
- Recibe:
    - `x`: tensor de entrada.
    - `sublayer`: funcion o modulo que procesa `x`.

### 6) `MultiHeadAttentionBlock`

Implementa atencion multi-cabeza (self-attention y cross-attention segun como se use).

- Proyecciones lineales:
    - `Wq`, `Wk`, `Wv`, `Wo`.
- Divide `d_model` en `h` cabezas, cada una de dimension `d_k = d_model / h`.
- Pasos:
    - proyectar `q`, `k`, `v`
    - separar en cabezas
    - calcular scores escalados
    - aplicar mascara (si existe)
    - `softmax`
    - mezclar con `value`
    - concatenar cabezas y aplicar `Wo`
- Guarda `attention_scores` para inspeccion.
- Entrada esperada en `forward`:
    - `q`, `k`, `v`: `(batch, seq_len, d_model)`
    - `mask`: forma compatible con broadcasting sobre scores.
- Salida: `(batch, seq_len, d_model)`.

### 7) `EncoderBlock`

Bloque base del encoder con dos subcapas.

- Subcapa 1: self-attention sobre la secuencia fuente.
- Subcapa 2: feed-forward.
- Cada subcapa esta envuelta con `ResidualConnection`.
- Entrada: `x` y `src_mask`.
- Salida: `x` transformado con misma forma `(batch, src_seq_len, d_model)`.

### 8) `Encoder`

Apila varios `EncoderBlock`.

- Recorre secuencialmente una lista de capas (`N` bloques).
- Aplica normalizacion final.
- Entrada: representaciones de fuente + mascara de fuente.
- Salida: memoria codificada del encoder.

### 9) `DecoderBlock`

Bloque base del decoder con tres subcapas.

- Subcapa 1: self-attention en target (normalmente con mascara causal).
- Subcapa 2: cross-attention contra salida del encoder.
- Subcapa 3: feed-forward.
- Todas con conexiones residuales.
- Entrada:
    - `x` (target embebido),
    - `encoder_output`,
    - `src_mask`,
    - `tgt_mask`.
- Salida: `(batch, tgt_seq_len, d_model)`.

### 10) `Decoder`

Apila varios `DecoderBlock`.

- Ejecuta los bloques en cascada.
- Aplica normalizacion final.
- Salida: representaciones finales del decoder para cada posicion target.

### 11) `ProjectionLayer`

Proyecta cada vector oculto al tamano del vocabulario destino.

- Usa `Linear(d_model -> vocab_size)`.
- Produce logits por token (antes de softmax).
- Entrada: `(batch, tgt_seq_len, d_model)`.
- Salida: `(batch, tgt_seq_len, tgt_vocab_size)`.

### 12) `Transformer`

Clase contenedora de alto nivel que une todos los bloques.

- `encode(src, src_mask)`:
    - embeddings de fuente
    - positional encoding de fuente
    - encoder
- `decode(encoder_output, src_mask, tgt, tgt_mask)`:
    - embeddings de target
    - positional encoding de target
    - decoder
- `project(x)`:
    - proyeccion final a logits de vocabulario

Separa claramente la etapa de codificacion, decodificacion y proyeccion.

### 13) `build_transformer(...)`

Funcion fabrica que crea un Transformer completo y lo inicializa.

- Construye embeddings y positional encodings.
- Crea `N` bloques de encoder y `N` bloques de decoder.
- Monta `Encoder`, `Decoder`, `ProjectionLayer` y `Transformer`.
- Inicializa pesos con Xavier uniforme para parametros matriciales.
- Devuelve un objeto listo para entrenar.

## Modificacion pedida: primera mitad con 32 heads y segunda mitad con 2 heads

Esta version ya esta adaptada en [modelo.py](modelo.py) para cumplir:

- Primeras capas: 32 cabezas de atencion.
- Ultimas capas: 2 cabezas de atencion.

### Donde se modifica

En la funcion `build_transformer(...)` de [modelo.py](modelo.py):

1. Se elimina el parametro global `h` (ya no hay un unico numero de heads para todas las capas).
2. Se define una lista por profundidad:
    - `heads_per_layer = [32] * (N // 2) + [2] * (N - N // 2)`
3. En los bucles de creacion de bloques, cada capa usa su propio `layer_heads`:
    - Encoder: `MultiHeadAttentionBlock(d_model, layer_heads, dropout)`
    - Decoder self-attention: `MultiHeadAttentionBlock(d_model, layer_heads, dropout)`
    - Decoder cross-attention: `MultiHeadAttentionBlock(d_model, layer_heads, dropout)`
4. Se anaden validaciones:
    - `N >= 2` para poder dividir en dos partes.
    - `d_model % layer_heads == 0` para cada capa.

### Como queda la distribucion

Si `N = 6`, entonces:

- Capas 0, 1, 2 -> 32 heads
- Capas 3, 4, 5 -> 2 heads

Si `N` es impar, la primera parte usa `N // 2` capas y la segunda usa el resto.

### Impacto en numero de parametros aprendidos

El numero de parametros aprendidos **no cambia** respecto a la implementacion original (con `h` fijo), siempre que `d_model` se mantenga igual.

Motivo:

- En cada bloque de atencion, las matrices entrenables siguen siendo:
  - `Wq`, `Wk`, `Wv`, `Wo`
- Cada una tiene tamano `d_model x d_model`.
- Cambiar el numero de heads solo cambia la particion interna (`d_k = d_model / h`), no el tamano total de esas matrices.

Por tanto, para cada bloque de atencion el total sigue siendo proporcional a `4 * d_model^2`, igual que antes.

## Nota de compatibilidad

Como `build_transformer(...)` ya no recibe `h`, cualquier llamada antigua con `h=...` debe eliminar ese argumento.

## Flujo del modelo

1. Tokens de entrada -> embeddings
2. Se suma codificacion posicional
3. Encoder procesa la secuencia fuente
4. Decoder procesa la secuencia objetivo con:
   - self-attention en target
   - cross-attention con salida del encoder
5. Proyeccion final al vocabulario destino

## Requisitos

- Python 3.9+
- PyTorch

Instalacion minima:

```bash
pip install torch
```


