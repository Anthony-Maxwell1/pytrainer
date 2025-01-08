import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from collections import defaultdict

class TextAI:
    def __init__(self):
        self.tokens = defaultdict(int)
        self.index_to_word = {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = 1.0

    def tokenize(self, text):
        tokens = text.split()
        tokenized_output = []
        for word in tokens:
            if word not in self.tokens:
                token_index = len(self.tokens)
                self.tokens[word] = token_index
                self.index_to_word[token_index] = word
            tokenized_output.append(self.tokens[word])
        return tokenized_output

    def prepare_data(self, data):
        X, y = [], []
        for prompt, response in data.items():
            prompt_tokens = self.tokenize(prompt)
            response_tokens = self.tokenize(response)
            X.append(prompt_tokens)
            y.append(response_tokens)
        return X, y

    def create_model(self, vocab_size, embedding_dim=128, hidden_dim=256):
        class Seq2SeqModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
                self.output = nn.Linear(hidden_dim, vocab_size)
                
            def forward(self, src, tgt=None, hidden=None):
                embedded_src = self.embedding(src)
                encoder_output, (hidden, cell) = self.encoder(embedded_src)
                
                if tgt is not None:  # Training mode
                    embedded_tgt = self.embedding(tgt)
                    decoder_output, _ = self.decoder(embedded_tgt, (hidden, cell))
                    output = self.output(decoder_output)
                    return output
                
                return hidden, cell  # For inference mode
        
        return Seq2SeqModel(vocab_size, embedding_dim, hidden_dim).to(self.device)

    def train(self, data, epochs=30, batch_size=1, learning_rate=0.001):
        X, y = self.prepare_data(data)
        vocab_size = len(self.tokens)
        
        self.model = self.create_model(vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            for idx in range(0, len(X), batch_size):
                X_batch = X[idx:idx+batch_size]
                y_batch = y[idx:idx+batch_size]
                
                # Pad sequences
                max_len_x = max(len(seq) for seq in X_batch)
                max_len_y = max(len(seq) for seq in y_batch)
                
                padded_x = [seq + [0] * (max_len_x - len(seq)) for seq in X_batch]
                padded_y = [seq + [0] * (max_len_y - len(seq)) for seq in y_batch]

                src = torch.tensor(padded_x, dtype=torch.long).to(self.device)
                tgt = torch.tensor(padded_y, dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(src, tgt)
                
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    def prompt(self, input_text, max_length=50, temperature=0.5):
        self.model.eval()
        
        # Tokenize input
        input_tokens = self.tokenize(input_text)
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        # Get initial hidden state
        hidden, cell = self.model(input_tensor)
        
        # Initialize with START token
        current_token = torch.tensor([[self.tokens.get("<START>", 0)]], dtype=torch.long).to(self.device)
        response_tokens = []
        
        # Track used tokens and their frequencies
        used_token_count = {}
        
        with torch.no_grad():
            for _ in range(max_length):
                embedded = self.model.embedding(current_token)
                output, (hidden, cell) = self.model.decoder(embedded, (hidden, cell))
                logits = self.model.output(output)
                
                # Apply temperature
                scaled_logits = logits / temperature
                
                # Clamp logits to prevent overflow/underflow
                scaled_logits = torch.clamp(scaled_logits, min=-10.0, max=10.0)
                
                # Penalize previously used tokens
                for token, count in used_token_count.items():
                    scaled_logits[0, -1, token] -= 2.0 + 0.1 * count  # Dynamic penalty based on usage
                
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs[0, -1], descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                nucleus_mask = cumsum_probs <= 0.9  # Use top 90% of probability mass
                
                sorted_probs[~nucleus_mask] = 0
                sorted_probs /= sorted_probs.sum()
                
                next_token_idx = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices[next_token_idx]
                
                if next_token.item() == self.tokens.get("<EOS>", 0):
                    break
                    
                token_word = self.index_to_word.get(next_token.item(), "")
                if token_word and token_word not in ["<START>", "<EOS>"]:
                    response_tokens.append(next_token.item())
                    # Update token usage count
                    used_token_count[next_token.item()] = used_token_count.get(next_token.item(), 0) + 1
                
                current_token = next_token.unsqueeze(0)
        
        response = " ".join([self.index_to_word[token] for token in response_tokens])
        return response

    def save(self, path):
        data = {
            "tokens": dict(self.tokens),
            "index_to_word": self.index_to_word,
            "model_weights": "model.pth"
        }
        torch.save(self.model.state_dict(), "model.pth")
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.tokens = defaultdict(int, data["tokens"])
            self.index_to_word = {int(k): v for k, v in data["index_to_word"].items()}
            
            vocab_size = len(self.tokens)
            self.model = self.create_model(vocab_size)
            self.model.load_state_dict(torch.load(data["model_weights"]))
            self.model.eval()