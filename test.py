import ollama

print("Checking Ollama...")

client = ollama.Client()
print(client.list())

print("Ollama is reachable")