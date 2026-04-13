# 🧠 Transformer – Preguntas Probables de Examen (con Respuestas)

---

## 1. Cambiar Pre-Norm → Post-Norm

### ❓ Pregunta

Modify the residual connection to use Post-Layer Normalization instead of Pre-Layer Normalization.

### 📍 Dónde

`ResidualConnection`

### ✅ Respuesta

Cambiar:

```python
return x + self.dropout(sublayer(self.norm(x)))
```

por:

```python
return self.norm(x + self.dropout(sublayer(x)))
```

---

## 2. Positional Encoding entrenable

### ❓ Pregunta

Modify the positional encoding so that it is learnable.

### 📍 Dónde

`PositionalEncoding`

### ✅ Respuesta

Cambiar:

```python
self.register_buffer('pe', pe)
```

por:

```python
self.pe = nn.Parameter(pe)
```

Eliminar:

```python
.requires_grad_(False)
```

---

## 3. Quitar scaling en embeddings

### ❓ Pregunta

Remove the scaling factor from input embeddings.

### 📍 Dónde

`InputEmbeddings`

### ✅ Respuesta

Cambiar:

```python
return self.embedding(x) * math.sqrt(self.d_model)
```

por:

```python
return self.embedding(x)
```

---

## 4. Cambiar ReLU por GELU

### ❓ Pregunta

Replace ReLU with GELU.

### 📍 Dónde

`FeedForwardBlock`

### ✅ Respuesta

Cambiar:

```python
torch.relu(...)
```

por:

```python
import torch.nn.functional as F
F.gelu(...)
```

---

## 5. Explicar la máscara

### ❓ Pregunta

Explain the role of the mask in attention.

### ✅ Respuesta

* Evita atender a padding
* Evita ver el futuro en el decoder
* Se aplica en:

```python
attention_scores.masked_fill_(mask == 0, -1e9)
```

* Sin máscara: el modelo vería información futura → incorrecto

---

## 6. Diferente número de capas encoder/decoder

### ❓ Pregunta

Allow different number of encoder and decoder layers.

### 📍 Dónde

`build_transformer`

### ✅ Respuesta

Separar:

```python
N_encoder
N_decoder
```

Y usar dos bucles distintos para construir encoder y decoder

---

## 7. Quitar Dropout

### ❓ Pregunta

Remove dropout from the model.

### ✅ Respuesta

* Eliminar `nn.Dropout`
* O usar:

```python
dropout=0
```

---

## 8. Multi-head → single-head

### ❓ Pregunta

Convert multi-head attention into single-head attention.

### ✅ Respuesta

Cambiar:

```python
h = 1
```

---

## 9. Dimensiones en MultiHeadAttention

### ❓ Pregunta

Explain the shape transformations.

### ✅ Respuesta

```
(batch, seq_len, d_model)
→ (batch, seq_len, h, d_k)
→ (batch, h, seq_len, d_k)
→ atención
→ (batch, h, seq_len, d_k)
→ (batch, seq_len, d_model)
```

---

## 10. Sampling en vez de argmax

### ❓ Pregunta

Modify decoding to sample instead of using argmax.

### ✅ Respuesta

Cambiar:

```python
torch.argmax(probs, dim=-1)
```

por:

```python
torch.multinomial(probs, 1)
```

Antes aplicar:

```python
probs = torch.softmax(logits, dim=-1)
```

---

## 11. Añadir temperatura

### ❓ Pregunta

Add temperature to sampling.

### ✅ Respuesta

```python
probs = torch.softmax(logits / temperature, dim=-1)
```

* temperature < 1 → más determinista
* temperature > 1 → más aleatorio

---

## 12. Diferentes heads por capa

### ❓ Pregunta

Use 32 heads in first half and 2 in second half.

### 📍 Dónde

`build_transformer`

### ✅ Respuesta

```python
for i in range(N):
    if i < N // 2:
        heads = 32
    else:
        heads = 2
```

Aplicar en encoder y decoder.

### ❗ Parámetros

NO cambian, porque dependen de `d_model`, no de `h`.

---

## 🧨 Pregunta Trampa

### ❓

Does increasing the number of heads increase the number of parameters?

### ✅ Respuesta

❌ NO

### 💡 Justificación

Las matrices:

```
Wq, Wk, Wv, Wo ∈ R^(d_model × d_model)
```

no dependen de `h`.

---

# 🆕 Más preguntas probables (nivel examen)

---

## 13. ¿Por qué dividir por sqrt(d_k)?

### ❓ Pregunta

Explain why the attention scores are scaled by 1/sqrt(d_k).

### ✅ Respuesta

* Evita valores grandes en el producto Q·K
* Sin escalado → softmax se satura
* Mejora estabilidad del entrenamiento

---

## 14. Quitar máscara del decoder

### ❓ Pregunta

What happens if we remove the target mask in the decoder?

### ✅ Respuesta

* El modelo ve tokens futuros
* Rompe el aprendizaje autoregresivo
* Generación incorrecta en inferencia

---

## 15. Compartir pesos embeddings

### ❓ Pregunta

Modify the model to share weights between input embeddings and projection layer.

### 📍 Dónde

Entre `InputEmbeddings` y `ProjectionLayer`

### ✅ Respuesta

```python
projection_layer.proj.weight = tgt_embed.embedding.weight
```

---

## 16. Cambiar dimensión del modelo

### ❓ Pregunta

What happens if we increase d_model?

### ✅ Respuesta

* Más capacidad del modelo
* Más parámetros (cuadrático)
* Más coste computacional

---

## 17. ¿Qué hace dropout en atención?

### ❓ Pregunta

Explain the role of dropout in attention.

### ✅ Respuesta

* Se aplica a los attention scores
* Reduce overfitting
* Hace el modelo más robusto

---

## 18. Eliminar feed-forward

### ❓ Pregunta

What happens if we remove the feed-forward block?

### ✅ Respuesta

* Menor capacidad de modelado
* Solo atención → menos expresividad

---

## 19. Cambiar softmax por otra función

### ❓ Pregunta

Replace softmax with another normalization function.

### ✅ Respuesta

* Se puede usar sigmoid o sparsemax
* Cambia la distribución de atención

---

## 20. Añadir capa extra en encoder

### ❓ Pregunta

Add one extra encoder layer.

### ✅ Respuesta

Aumentar N en `build_transformer`

---

## 21. Explicar cross-attention

### ❓ Pregunta

Explain cross-attention in the decoder.

### ✅ Respuesta

* Query → decoder
* Key/Value → encoder
* Permite al decoder usar información del input

---

## 22. ¿Qué pasa si h no divide d_model?

### ❓ Pregunta

What happens if d_model is not divisible by h?

### ✅ Respuesta

* Error en el código (assert)
* No se puede dividir en heads iguales

---

## 23. Cambiar orden encoder-decoder

### ❓ Pregunta

What happens if we remove the encoder and use only the decoder?

### ✅ Respuesta

* Se convierte en un modelo tipo GPT
* Solo usa contexto previo

---

## 24. Añadir batch normalization

### ❓ Pregunta

Replace LayerNorm with BatchNorm.

### ✅ Respuesta

* Cambiar clase LayerNormalization
* No recomendable en secuencias

---

## 25. ¿Qué hace contiguous()?

### ❓ Pregunta

Explain the role of contiguous() in tensor reshaping.

### ✅ Respuesta

* Asegura memoria continua
* Necesario antes de view()

---

# 🚀 Resumen Final

Dominar:

* MultiHeadAttention
* Mask
* Residual + LayerNorm
* build_transformer

---

Dominar:

* MultiHeadAttention
* Mask
* Residual + LayerNorm
* build_transformer

---
