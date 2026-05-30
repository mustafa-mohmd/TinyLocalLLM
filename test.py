import ollama

print("Checking Ollama...")

client = ollama.Client()
print(client.list())
#end case
print("Ollama is reachable")
