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

    def tokenize(self, text, nearestPunctuation=False, runtime=False):
        tokens = text.split()
        tokenized_output = []
        for word in tokens:
            if word not in self.tokens:
                if runtime:
                    if nearestPunctuation:
                        # Check for variations of the word with different punctuations
                        variations = [word, word.lower(), word.capitalize()] + [word.rstrip('.,!?') + punct for punct in ['.', ',', '!', '?']] + [word.rstrip('.,!?').lower() + punct for punct in ['.', ',', '!', '?']] + [word.rstrip('.,!?').capitalize() + punct for punct in ['.', ',', '!', '?']]
                        # print(variations)
                        for variation in variations:
                            if variation in self.tokens:
                                token_index = self.tokens[variation]
                                tokenized_output.append(token_index)
                                # print(f"Used variation: {variation}")
                                break
                else:
                    token_index = len(self.tokens)
                    self.tokens[word] = token_index
                    self.index_to_word[token_index] = word
                    tokenized_output.append(self.tokens[word])
            else:
                tokenized_output.append(self.tokens[word])
        # print(tokenized_output)
        if len(tokenized_output) <= 0: return False
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
                
                # Initialize weights with Xavier/Glorot initialization
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                
            def forward(self, src, tgt=None, hidden=None):
                embedded_src = self.embedding(src)
                
                # Add gradient clipping at the embedding level
                embedded_src = torch.clamp(embedded_src, -100, 100)
                
                encoder_output, (hidden, cell) = self.encoder(embedded_src)
                
                if tgt is not None:  # Training mode
                    embedded_tgt = self.embedding(tgt)
                    embedded_tgt = torch.clamp(embedded_tgt, -100, 100)
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
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # Added LR scheduler

        for epoch in range(epochs):
            total_loss = 0
            self.model.train()  # Explicitly set training mode
            
            for idx in range(0, len(X), batch_size):
                X_batch = X[idx:idx+batch_size]
                y_batch = y[idx:idx+batch_size]
                
                max_len_x = max(len(seq) for seq in X_batch)
                max_len_y = max(len(seq) for seq in y_batch)
                
                padded_x = [seq + [0] * (max_len_x - len(seq)) for seq in X_batch]
                padded_y = [seq + [0] * (max_len_y - len(seq)) for seq in y_batch]

                src = torch.tensor(padded_x, dtype=torch.long).to(self.device)
                tgt = torch.tensor(padded_y, dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(src, tgt)
                
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    print(f"Warning: Invalid loss detected in epoch {epoch+1}. Skipping batch.")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X) // batch_size)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)

    def prompt(self, input_text, max_length=50, temperature=0.7, nearestPunctuation=False):
        self.model.eval()
        
        input_tokens = self.tokenize(input_text, nearestPunctuation, True)
        if input_tokens == False: return ""
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
        
        hidden, cell = self.model(input_tensor)
        current_token = torch.tensor([[self.tokens.get("<START>", 0)]], dtype=torch.long).to(self.device)
        response_tokens = []
        used_token_count = {}
        
        with torch.no_grad():
            for _ in range(max_length):
                try:
                    embedded = self.model.embedding(current_token)
                    embedded = torch.clamp(embedded, -100, 100)  # Clamp embeddings
                    
                    output, (hidden, cell) = self.model.decoder(embedded, (hidden, cell))
                    logits = self.model.output(output)
                    
                    # Apply temperature and clamp logits
                    scaled_logits = torch.clamp(logits / temperature, min=-10.0, max=10.0)
                    
                    # Apply token penalties
                    for token, count in used_token_count.items():
                        penalty = min(2.0 + 0.1 * count, 5.0)  # Cap the maximum penalty
                        scaled_logits[0, -1, token] -= penalty
                    
                    # Stable softmax with numerical stability improvements
                    max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
                    exp_logits = torch.exp(scaled_logits - max_logits)
                    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
                    
                    # Ensure valid probabilities
                    probs = torch.nan_to_num(probs, nan=1e-6, posinf=1-1e-6, neginf=1e-6)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    # Nucleus sampling
                    sorted_probs, sorted_indices = torch.sort(probs[0, -1], descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                    nucleus_mask = cumsum_probs <= 0.9
                    
                    sorted_probs[~nucleus_mask] = 0
                    # Ensure we have at least one valid token
                    if sorted_probs.sum() == 0:
                        sorted_probs[0] = 1.0
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    
                    next_token_idx = torch.multinomial(sorted_probs, 1)
                    next_token = sorted_indices[next_token_idx]
                    
                except RuntimeError as e:
                    print(f"Warning: RuntimeError in generation: {e}")
                    break
                
                if next_token.item() == self.tokens.get("<EOS>", 0):
                    break
                    
                token_word = self.index_to_word.get(next_token.item(), "")
                if token_word and token_word not in ["<START>", "<EOS>"]:
                    response_tokens.append(next_token.item())
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