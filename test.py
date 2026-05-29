import ollama

print("Checking Ollama...")

client = ollama.Client()
print(client.list())
#end case no
print("Ollama is reachable")