def simple_tokenizer(text):
    return text.lower().split()

def tokenize_texts(texts, model_type, vocab, transformer_tokenizer=None):
    if model_type == "transformer" and transformer_tokenizer is not None:
        batch = transformer_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return batch['input_ids'], batch.get('attention_mask', None)
    else:
        # Simple tokenization and conversion to ids for RNN/LSTM/GRU
        import torch
        max_len = max(len(text.lower().split()) for text in texts)
        input_ids = []
        for text in texts:
            tokens = text.lower().split()
            ids = [vocab.get(tok, vocab.get('<UNK>', vocab.get('<unk>', 1))) for tok in tokens]
            # Pad to max_len
            ids += [vocab.get('<PAD>', vocab.get('<pad>', 0))] * (max_len - len(ids))
            input_ids.append(ids)
        input_ids = torch.tensor(input_ids)
        return input_ids, None
