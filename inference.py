import torch
from modelo25Sol import build_transformer

# ============================================================
# INFERENCIA: GREEDY (ARGMAX) vs SAMPLING
# ============================================================

def decode_greedy(model, src, src_mask, max_len, start_token, device):
    """
    Decodificacion GREEDY: selecciona el token con probabilidad maxima (argmax).
    Output determinista.
    """
    model.eval()
    with torch.no_grad():
        # Codifica la fuente
        encoder_output = model.encode(src, src_mask)
        
        # Inicia la secuencia con el token de inicio
        decoder_input = torch.tensor([[start_token]], dtype=torch.long, device=device)
        
        # Genera tokens hasta max_len o hasta token de fin
        for _ in range(max_len):
            # Mascara causal para el decoder (simplificada, asumo que ya existe)
            tgt_mask = torch.ones((decoder_input.shape[0], 1, decoder_input.shape[1], decoder_input.shape[1]), device=device)
            
            # Forward pass del decoder y proyeccion
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            logits = model.project(decoder_output)  # (batch, seq_len, vocab_size)
            
            # GREEDY: Argmax del ultimo token
            # Toma solo el ultimo token predicho y su probabilidad maxima
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            
            # Concatena al decoder_input para la siguiente iteracion
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        return decoder_input


def decode_sampling(model, src, src_mask, max_len, start_token, device, temperature=1.0):
    """
    Decodificacion con SAMPLING: muestrea segun la distribucion de probabilidades.
    Output estocastico (variable segun la muestra).
    
    Args:
        temperature: controla la "temperatura" de la distribucion.
                   - temperature < 1: distribucion mas concentrada (menos diversidad).
                   - temperature = 1: distribucion normal.
                   - temperature > 1: distribucion mas dispersa (mas diversidad).
    """
    model.eval()
    with torch.no_grad():
        # Codifica la fuente
        encoder_output = model.encode(src, src_mask)
        
        # Inicia la secuencia con el token de inicio
        decoder_input = torch.tensor([[start_token]], dtype=torch.long, device=device)
        
        # Genera tokens hasta max_len o hasta token de fin
        for _ in range(max_len):
            # Mascara causal para el decoder
            tgt_mask = torch.ones((decoder_input.shape[0], 1, decoder_input.shape[1], decoder_input.shape[1]), device=device)
            
            # Forward pass del decoder y proyeccion
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            logits = model.project(decoder_output)  # (batch, seq_len, vocab_size)
            
            # SAMPLING: Muestrea segun la distribucion
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Aplicar temperatura a los logits
            scaled_logits = next_token_logits / temperature
            
            # Convertir logits a probabilidades con softmax
            probs = torch.softmax(scaled_logits, dim=-1)  # (batch, vocab_size)
            
            # Muestrear un token segun esas probabilidades
            # torch.multinomial: sample sin reemplazo de una distribucion multinomial
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Concatena al decoder_input para la siguiente iteracion
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        return decoder_input


def decode_top_k(model, src, src_mask, max_len, start_token, device, top_k=10):
    """
    Decodificacion Top-K: muestrea solo de los K tokens con mas probabilidad (hibrido).
    Mitiga tokens muy improbables pero mantiene diversidad.
    """
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        decoder_input = torch.tensor([[start_token]], dtype=torch.long, device=device)
        
        for _ in range(max_len):
            tgt_mask = torch.ones((decoder_input.shape[0], 1, decoder_input.shape[1], decoder_input.shape[1]), device=device)
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            logits = model.project(decoder_output)
            
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Top-K: mantener solo los top_k logits, el resto a -inf
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Renormalizar probabilidades entre los top_k
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            # Muestrear de los top_k
            sampled_idx = torch.multinomial(top_k_probs, num_samples=1)  # (batch, 1)
            next_token = torch.gather(top_k_indices, 1, sampled_idx)  # (batch, 1)
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        return decoder_input


# ============================================================
# COMPARATIVA: GREEDY vs SAMPLING
# ============================================================

if __name__ == "__main__":
    # Parametros
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    N = 6
    
    # Construir modelo
    model = build_transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        src_seq_len=64,
        tgt_seq_len=64,
        d_model=d_model,
        N=N,
        dropout=0.1,
        d_ff=2048
    ).to(device)
    
    # Dummy input
    batch_size = 1
    src = torch.randint(0, 10000, (batch_size, 32)).to(device)
    src_mask = torch.ones((batch_size, 1, 1, 32)).to(device)
    
    # Generar con GREEDY
    print("=" * 60)
    print("GREEDY DECODING (argmax, determinista)")
    print("=" * 60)
    output_greedy = decode_greedy(model, src, src_mask, max_len=20, start_token=1, device=device)
    print(f"Output shape: {output_greedy.shape}")
    print(f"Output tokens: {output_greedy}")
    
    # Generar con SAMPLING
    print("\n" + "=" * 60)
    print("SAMPLING DECODING (muestreo aleatorio, estocastico)")
    print("=" * 60)
    output_sample1 = decode_sampling(model, src, src_mask, max_len=20, start_token=1, device=device, temperature=1.0)
    print(f"Output shape: {output_sample1.shape}")
    print(f"Output tokens (muestra 1): {output_sample1}")
    
    output_sample2 = decode_sampling(model, src, src_mask, max_len=20, start_token=1, device=device, temperature=1.0)
    print(f"Output tokens (muestra 2): {output_sample2}")
    print(f"Son diferentes (estocastico): {not torch.equal(output_sample1, output_sample2)}")
    
    # Con temperatura mas alta (mas diversidad)
    print("\nSampling con temperature=2.0 (mas diversidad):")
    output_sample_hot = decode_sampling(model, src, src_mask, max_len=20, start_token=1, device=device, temperature=2.0)
    print(f"Output tokens: {output_sample_hot}")
    
    # Con Top-K
    print("\n" + "=" * 60)
    print("TOP-K SAMPLING (hibrido)")
    print("=" * 60)
    output_topk = decode_top_k(model, src, src_mask, max_len=20, start_token=1, device=device, top_k=10)
    print(f"Output tokens: {output_topk}")
