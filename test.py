import ollama
#justtotest
print("Checking Ollama...")

client = ollama.Client()
print(client.list())
#end case
print("Ollama is reachable")
