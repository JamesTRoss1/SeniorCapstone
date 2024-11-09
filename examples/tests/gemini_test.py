import pathlib
import textwrap
import os 
import google.generativeai as genai

genai.configure(api_key=YOUR_API_KEY)

model = genai.GenerativeModel('gemini-pro')

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
print()

response = model.generate_content("Hello")

print(str(response.text))