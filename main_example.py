from flask import Flask, request, jsonify
import os
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

app = Flask(__name__)

PROMPT_TEMPLATE = """
context: {context}
history: {history}
input: {input}

Generate an answer to an input that takes into account context and history.
answer:
"""

class CustomPromptTemplate:
    def __init__(self, template, context):
        self.template = template
        self.context = context

    def format(self, **kwargs):
        context = kwargs.get('context', '')
        return self.template.format(context=context, **kwargs)

PROMPT = CustomPromptTemplate(
    template=PROMPT_TEMPLATE,
    context="""
     You are a chatbot who needs to get information. 
     The information you need to get from the user should get favorite season, favorite scent, today's emotion, The atmosphere user's want. 
     If the user only tells you some information during the conversation 
     you should speak so that you can get answers from the user by asking questions to get other information
    """
)

def get_response(formatted_prompt):
    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": formatted_prompt}],
    temperature=0.7)
    return response.choices[0].message.content

@app.route("/api/chats", methods=['POST'])
def chat():
    data = request.json
    input_text = data['input']
    extraInfo = data.get('extraInfo', {})

    history = (
        f"You are a chatbot who needs to get information."
        f"  The information you need to get from the user should get favorite season, favorite scent, today's emotion, "
        f" The atmosphere user's want."
        f" If the user only tells you some information during the conversation, "
        f" you should speak so that you can get answers from the user by asking questions to get other information"
        f" Change your tone to suit your personality."
        f" If the user answered a question, do not ask the question again"
        f" When responding, use lots of appropriate emojis."
        f" Please always answer in Korean."
        f" Please always answer in a casual tone."
        f" Ask the user a question, Don't tell me what you like"
        f" Talk to me like a 5-year-old."
        # f" User's name is {extraInfo.get('nickname', 'Unknown')}, "
        # f" User's gender is {extraInfo.get('gender', 'Unknown')}, "
        # f" User's age is {extraInfo.get('age', 'Unknown')}, "
        # f" User's favorite season is {extraInfo.get('season', 'Unknown')}, "
        # f" User's likedScents is {extraInfo.get('likedScents', 'Unknown')}, "
        # f" User's dislikedScents is {extraInfo.get('dislikedScents', 'Unknown')}, "
    )

    formatted_prompt = PROMPT.format(history=history, input=input_text)
    response_text = get_response(formatted_prompt)

    replacements = {
        "이야": "이애오",
        "이에요": "이애오",
        "세요": "새오",
        "게요": "게오",
        "어요": "어오",
        "해요": "해오",
        "이예요": "이애오",
        "요": "오",
        "아요": "아오",
    }

    for key, value in replacements.items():
        response_text = response_text.replace(key, value)

    return jsonify({'input': input_text, 'response': response_text})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)