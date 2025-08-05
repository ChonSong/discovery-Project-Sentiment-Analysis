import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.transformer import TransformerModel
from utils import tokenize_texts, simple_tokenizer
from train import train_model
from evaluate import evaluate_model

# 1. Load Data
df = pd.read_csv("exorde_raw_sample.csv")
# Assume 'text' column contains your texts, and 'label' contains your sentiment/emotion label

# 2. Downstream Processing (cleaning, lowercasing, etc.)
df = df.dropna(subset=['text', 'label'])
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(int).tolist()

# 3. Build Vocabulary (for RNN/LSTM/GRU)
all_tokens = [tok for txt in texts for tok in simple_tokenizer(txt)]
vocab = {'<pad>':0, '<unk>':1}
for tok in set(all_tokens):
    if tok not in vocab:
        vocab[tok] = len(vocab)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 5. Tokenize
def prepare_data(texts, labels, model_type, vocab):
    input_ids, _ = tokenize_texts(texts, model_type, vocab)
    labels = torch.tensor(labels)
    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

train_loader = prepare_data(X_train, y_train, "rnn", vocab)   # change "rnn" to your model_type
test_loader = prepare_data(X_test, y_test, "rnn", vocab)

# 6. Model Selection
model_type = "rnn"  # or "lstm", "gru", "transformer"
model_dict = {
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "transformer": TransformerModel,
}
params = dict(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_dim=64,
    num_classes=len(set(labels)),
    num_heads=2,
    num_layers=2
)
if model_type != "transformer":
    params.pop("num_heads")
    params.pop("num_layers")
model = model_dict[model_type](**params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 7. Training and Evaluation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
metric_fn = lambda y_true, y_pred: (y_true == y_pred).float().mean().item()

# Train
train_model(model, train_loader, optimizer, loss_fn, device)
# Evaluate
acc = evaluate_model(model, test_loader, metric_fn, device)
print(f"Test Accuracy: {acc:.4f}")
