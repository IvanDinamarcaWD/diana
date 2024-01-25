import random
from typing import Dict, Any, Iterator
from diana import newPrompt
import asyncio
def generate_random_data() -> Iterator[Dict[str, Any]]:
    while True:
        yield {
            'key1': random.choice([1, 2, 3, 'a', 'b', 'c', True, False]),
            'key2': random.randint(1, 100),
            'key3': random.uniform(0.0, 1.0),
            # Add more keys and random value generation as needed
        }

async def _generate(answer: str):
	chunk_size = 10
	num_chunks = (len(answer) + chunk_size - 1) // chunk_size
	for i in range(num_chunks):
		chunk_start = i * chunk_size
		chunk_end = (i + 1) * chunk_size
		chunk = answer[chunk_start:chunk_end]
		response_body = f"{chunk}"
		print(response_body)
		yield response_body.encode()
		await asyncio.sleep(0.1)

# Example usage:
if __name__ == "__main__":
    data_generator = newPrompt('hola, responde en dos palabras')

    _generate(data_generator)
    #for i in data_generator:
    #    print(next(i))