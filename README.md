> [!WARNING]
> This is a project that was created in one or two afternoons as a learning experience and gateway into AI. Don't use it! Just use pytorch.

# Pytrainer

AI made Easy.

## Tutorials

### Train and Save a Model

```python
from pytrainer.text import TextAI

# Create an instance of TextAI
ai = TextAI()

# Define some simple training data (input-output pairs)
training_data = {
    "Hello": "Hi there!",
    "How are you?": "I'm good, thank you!",
    "What's your name?": "I am TextAI."
}

# Train the model
ai.train(training_data, epochs=10, batch_size=1, learning_rate=0.001)

# Save the trained model
ai.save('trained_model.json')
```

### Load and Prompt the Model

```python
from pytrainer.text import TextAI

# Create a new instance of TextAI and load the trained model
ai_new = TextAI()
ai_new.load('trained_model.json')

# Generate a response to a new prompt
response = ai_new.prompt("What's your name?")
print(response)
```

## Classes

### `TextAI`

#### from pytrainer.text import TextAI

#### `train(self, data, epochs=10, batch_size=1, learning_rate=0.001)`

- **Arguments**:
  - `data`: A dictionary of input-output pairs for training (required).
  - `epochs`: The number of training epochs (default is 10).
  - `batch_size`: The batch size used during training (default is 1).
  - `learning_rate`: The learning rate for the optimizer (default is 0.001).

#### `prompt(self, input_text, max_length=50, temperature=0.5)`

- **Arguments**:
  - `input_text`: The text prompt to generate a response for (required).
  - `max_length`: The maximum length of the generated response (default is 50).
  - `temperature`: Controls the randomness of the generated text (default is 0.7).
  - `nearestPunctuation`: Should a word be rounded to the same word with different punctuation, if that word is not tokenized? (defualt is False)

#### `save(self, path)`

- **Arguments**:
  - `path`: The file path to save the model data (required).

#### `load(self, path)`

- **Arguments**:
  - `path`: The file path from which to load the model data (required).

#### Known Issues:

(Fixed errors will be removed next update, and moved to change log in github.)

- Exploding gradients during training of over 30 epochs, resulting in NaN errors when prompting. **FIXED**
- When prompting, using an un-tokenized word will fail. **FIXED**
  - Note: There is a new option to map un-tokenized words to the same word with different punctation, if one was tokenized. Otherwise, it will return nothing.

#### Planned Features:

- Individual classes for Tokenizers, allowing them to be customized.
- Option for built-in tokenizers, so the trained model is backward compatible with pytorch.
