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

# 🧩 Más preguntas SOLO de modificar código (muy probables)

---

## 26. Quitar bias en todas las capas lineales

### ❓ Pregunta

Modify the model to remove bias from all linear layers.

### 📍 Dónde

En todos los `nn.Linear`

### ✅ Respuesta

Cambiar:

```python
nn.Linear(in, out)
```

por:

```python
nn.Linear(in, out, bias=False)
```

---

## 27. Añadir dropout en embeddings

### ❓ Pregunta

Add dropout after embeddings.

### 📍 Dónde

`InputEmbeddings`

### ✅ Respuesta

En `__init__`:

```python
self.dropout = nn.Dropout(dropout)
```

En `forward`:

```python
return self.dropout(self.embedding(x) * math.sqrt(self.d_model))
```

---

## 28. Cambiar activación por LeakyReLU

### ❓ Pregunta

Replace ReLU with LeakyReLU.

### 📍 Dónde

`FeedForwardBlock`

### ✅ Respuesta

```python
torch.nn.functional.leaky_relu(...)
```

---

## 29. Añadir normalización final al decoder

### ❓ Pregunta

Ensure decoder output is normalized before projection.

### 📍 Dónde

`Decoder`

### ✅ Respuesta

Ya existe, pero si no:

```python
return self.norm(x)
```

---

## 30. Quitar residual connections

### ❓ Pregunta

Remove residual connections.

### 📍 Dónde

`ResidualConnection`

### ✅ Respuesta

```python
return self.dropout(sublayer(self.norm(x)))
```

---

## 31. Usar misma atención en encoder y decoder

### ❓ Pregunta

Share attention weights between encoder and decoder.

### 📍 Dónde

`build_transformer`

### ✅ Respuesta

Crear una instancia y reutilizarla:

```python
shared_attention = MultiHeadAttentionBlock(...)
```

---

## 32. Cambiar inicialización de pesos

### ❓ Pregunta

Replace Xavier initialization with normal initialization.

### 📍 Dónde

Final de `build_transformer`

### ✅ Respuesta

```python
nn.init.normal_(p, mean=0, std=0.02)
```

---

## 33. Añadir capa extra en FFN

### ❓ Pregunta

Add an extra linear layer to the feed-forward block.

### 📍 Dónde

`FeedForwardBlock`

### ✅ Respuesta

Añadir:

```python
self.linear_3 = nn.Linear(d_ff, d_ff)
```

Y forward:

```python
return self.linear_2(self.dropout(torch.relu(self.linear_3(torch.relu(self.linear_1(x))))))
```

---

## 34. Cambiar orden Dropout

### ❓ Pregunta

Apply dropout before linear transformation.

### 📍 Dónde

`FeedForwardBlock`

### ✅ Respuesta

Mover dropout antes de linear_2

---

## 35. Quitar cross-attention

### ❓ Pregunta

Remove cross-attention from decoder.

### 📍 Dónde

`DecoderBlock`

### ✅ Respuesta

Eliminar:

```python
self.cross_attention_block
```

y su uso en forward

---

## 36. Añadir máscara al encoder

### ❓ Pregunta

Ensure encoder uses padding mask.

### 📍 Dónde

`EncoderBlock`

### ✅ Respuesta

Ya se pasa como `src_mask`, asegurarse de usarlo en self-attention

---

## 37. Cambiar view por reshape

### ❓ Pregunta

Replace view with reshape.

### 📍 Dónde

MultiHeadAttention

### ✅ Respuesta

```python
tensor.reshape(...)
```

---

## 38. Eliminar softmax en atención

### ❓ Pregunta

Remove softmax from attention.

### 📍 Dónde

`attention`

### ✅ Respuesta

Eliminar:

```python
.softmax(dim=-1)
```

---

## 39. Añadir capa final de activación

### ❓ Pregunta

Add activation after projection.

### 📍 Dónde

`ProjectionLayer`

### ✅ Respuesta

```python
torch.softmax(self.proj(x), dim=-1)
```

---

## 40. Cambiar máscara por -inf

### ❓ Pregunta

Use -inf instead of -1e9.

### 📍 Dónde

`attention`

### ✅ Respuesta

```python
float('-inf')
```

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

Dominar:

* MultiHeadAttention
* Mask
* Residual + LayerNorm
* build_transformer

---
