from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

prompt = "10 * 10"
response = client.chat.completions.create(
    model='phi3',
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)