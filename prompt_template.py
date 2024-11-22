from langchain_core.prompts import PromptTemplate

def create_prompt():
    prompt = PromptTemplate.from_template(
             """"You are a perfume creation assistant, and your name is Scentasy.
    It means to make your dreams come true with scents.
    The user prefers to be called {nickname}님.
    Your goal is to help users create personalized perfume recipes by gently extracting information about their mood, current situation, feelings, weather, season, location, and companionship in a natural and empathetic way.
    Ask detailed questions to understand why they feel a certain way or what’s happening in their life today.
    Aqua, green, 베르가뭇, mint, fig, lemon, peach, orange, grapefruit, magnolia, rose, freesia, muget, ocean, blackcurrant, black pepper, rosemary, cedarwood, sandalwood, amber, white musk, vanilla, leather, hinoki, patchouli, and frankincense are the only scents you can mention if you recommend scents.
    Ask one question at a time, ensuring a soft and friendly tone.
    Use emojis to make the conversation more engaging and warm.
    Don’t recommend specific scents too early in the conversation. Instead, ask for more information about the user’s situation and preferences.
    When enough information is gathered, summarize the user’s input, and only then suggest relevant scents based on the gathered data.
    Don’t suggest specific ingredient combinations like '레몬, 우디한 느낌의 샌달우드, 그리고 은은한 바닐라' unless the user specifically asks.
    When enough information is collected, smoothly transition to wrapping up the conversation by asking, '모든 정보를 다 모았어요! 이제 당신만을 위한 향수를 만들어볼까요? 😊'
    Answer in Korean."


# Previous Chat History:
{chat_history}

# Question:
{question}

# Context:
{context}

# Answer:"""
    )
    return prompt

# def create_title_description_prompt():
#     return PromptTemplate.from_template(
#         """Based on the following conversation and predicted perfume characteristics, generate a creative title and description for the perfume.

#         Conversation:
#         {conversation_text}

#         Predicted Notes: {notes_str}
#         Predicted Accords: {accords_str}

#         Instructions:
#         1. The title should be short and appealing.
#         2. The description should creatively express the scent's feeling and characteristics, using the predicted notes and accords.
#         3. Make sure to consider the mood and details from the conversation when writing the description.
#         4. Please respond in Korean.

#         Example description:
#         제목: 자신감의 순간
#         설명: 중요한 순간에 자신감을 높여주는 우아하고 정돈된 향입니다. 상쾌한 시트러스 노트로 시작해 은은한 백합과 자스민의 플로럴 어코드가 지적이고 우아한 분위기를 연출합니다. 베이스로는 따뜻한 샌달우드와 머스크가 안정감 있는 마무리를 만들어 긴장 속에서도 차분함과 자신감을 유지하게 해줍니다.

#         Generate the title and description:"""
#     )

def create_title_description_prompt():
    return PromptTemplate.from_template(
        """Based on the following conversation and predicted perfume characteristics, please create a creative title and description.

        Conversation:
        {conversation_text}

        Predicted Notes: {notes_str}
        Predicted Accords: {accords_str}

        Instructions:
        1. Both the title and description must be included.
        2. The title should be short and appealing.
        3. The description should creatively express the scent's feeling and characteristics, using the predicted notes and accords.
        4. Reflect the mood and details from the conversation in the description.
        5. Both the title and description must be included.
        6. Please respond in Korean.

        Example description:
        제목: 자신감의 순간
        설명: 중요한 순간에 자신감을 높여주는 우아하고 정돈된 향입니다. 상쾌한 시트러스 노트로 시작해 은은한 백합과 자스민의 플로럴 어코드가 지적이고 우아한 분위기를 연출합니다. 베이스로는 따뜻한 샌달우드와 머스크가 안정감 있는 마무리를 만들어 긴장 속에서도 차분함과 자신감을 유지하게 해줍니다.
        
        The description should not be too long
        Generate the title and description:"""
    )
