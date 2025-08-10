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
print(f"Loaded dataset with columns: {list(df.columns)}")

# 2. Downstream Processing (cleaning, lowercasing, etc.)
# Use the correct column names from the dataset
text_col = 'original_text'
sentiment_col = 'sentiment'

df = df.dropna(subset=[text_col, sentiment_col])
texts = df[text_col].astype(str).tolist()

# Convert continuous sentiment scores to categorical labels
def categorize_sentiment(score):
    """Convert continuous sentiment score to categorical label."""
    try:
        score = float(score)
        if score < -0.1:
            return 0  # Negative
        elif score > 0.1:
            return 2  # Positive 
        else:
            return 1  # Neutral
    except:
        return 1  # Default to neutral for invalid scores

labels = [categorize_sentiment(s) for s in df[sentiment_col].tolist()]
print(f"Processed {len(texts)} samples")
print(f"Label distribution: Negative={labels.count(0)}, Neutral={labels.count(1)}, Positive={labels.count(2)}")

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
    num_classes=3,  # Negative, Neutral, Positive
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

print(f"\nTraining {model_type} model...")
print("=" * 40)

# Train for multiple epochs with validation
from train import train_model_epochs
history = train_model_epochs(model, train_loader, test_loader, optimizer, loss_fn, device, num_epochs=10)

# Final evaluation
final_accuracy = evaluate_model(model, test_loader, None, device)
print(f"\nFinal {model_type.upper()} Test Accuracy: {final_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), f"trained_{model_type}_model.pt")
print(f"Model saved as: trained_{model_type}_model.pt")
