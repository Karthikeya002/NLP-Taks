import re, requests, torch
import torch.nn as nn
from bs4 import BeautifulSoup
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class ChunkerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x): 
        lstm_out = self.lstm(x)[0]
        last_time_step_out = lstm_out[:, -1, :]
        return torch.sigmoid(self.fc(last_time_step_out))

def fetch_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text_content = ' '.join(p.get_text() for p in soup.find_all('p'))
    return re.sub(r'\s+', ' ', text_content).strip()

def preprocess(text):
    tok = Tokenizer(num_words=5000)
    tok.fit_on_texts([text])
    sequences = tok.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
    return padded_sequences, tok

def segment_text(url):
    full_text = fetch_text(url)
    seq, tok = preprocess(full_text)
    
    return full_text.split('. ')[:5]

print("Extracted Chunks:")
try:
    print(*segment_text("https://www.geeksforgeeks.org/nlp/natural-language-processing-nlp-tutorial/"), sep="\n")
except Exception as e:
    print(f"An error occurred during execution: {e}")