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
  - `temperature`: Controls the randomness of the generated text (default is 0.5).

#### `save(self, path)`

- **Arguments**:
  - `path`: The file path to save the model data (required).

#### `load(self, path)`

- **Arguments**:
  - `path`: The file path from which to load the model data (required).
