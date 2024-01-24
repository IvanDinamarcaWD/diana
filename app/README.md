`pip install -r requeriments.txt`
`uvicorn main:app --reload`

`curl -X POST -H "Content-Type: application/json" -d '{"message": "¿Cómo te llamas?"}' http://127.0.0.1:8000/prompt`