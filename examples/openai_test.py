#using openai's API is a very good option, although it is not a free option and one pays per query as opposed to a monthly plan 
#the API also supports text-to-speech so we can easily provide both a text option and a speech option and there are six voices to choose from
#some further playing around will be needed to modify the text to provide emotional range in the tone as there is not a direct way to affect this besides the statement itself
#one of the biggest hurdles will be crafting the perfectly tuned input into the model (prompt engineering)
#could either use gpt-4 turbo or gpt-3.5 turbo though gpt-4 is about 20 times as expensive per query but is a more powerful and capable model
#an additional downside is that since this solution uses a API endpoint, it will require an internet connection 

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a response generator that will modify a given statement depending on the most prevalent emotion that the audience is feeling."}
  ]
)

print(completion.choices[0].message)