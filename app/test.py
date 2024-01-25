import random
from typing import Dict, Any, Iterator
from diana import newPrompt

def generate_random_data() -> Iterator[Dict[str, Any]]:
    while True:
        yield {
            'key1': random.choice([1, 2, 3, 'a', 'b', 'c', True, False]),
            'key2': random.randint(1, 100),
            'key3': random.uniform(0.0, 1.0),
            # Add more keys and random value generation as needed
        }

# Example usage:
if __name__ == "__main__":
    data_generator = newPrompt('hola, responde en dos palabras')

    for _ in data_generator:
        print(next(data_generator))