from transformers import AutoTokenizer, AutoModel
import torch

def get_bert_embeddings(text, model_name='bert-base-uncased', pooling='cls', max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state.squeeze(0)
    
    if pooling == 'cls':
        pooled = last_hidden_state[0]
    elif pooling == 'mean':
        pooled = torch.mean(last_hidden_state, dim=0)
    elif pooling == 'max':
        pooled, _ = torch.max(last_hidden_state, dim=0)
    else:
        raise ValueError("please choose pooling=='cls', 'mean' or 'max'")
    
    return pooled.numpy()

if __name__ == "__main__":
    text = "This is a sample sentence for BERT embedding."
    embedding = get_bert_embeddings(text)
    print(f"Embedding Vector Shape: {embedding.shape}")
    print(embedding)