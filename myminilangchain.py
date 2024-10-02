from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI
from gemini_api import GeminiClient  # Assuming the package exists

# Initialize APIs
openai_api = OpenAI(api_key="your_openai_key")
gemini_client = GeminiClient(api_key="your_gemini_api_key")

# Create prompt templates
prompt_template = PromptTemplate(
    input_variables=["message", "sentiment"],
    template="The user feels {sentiment}. Based on this, respond to the message: {message}."
)

# Now we candefine the chain here with OpenAI or Gemini
llm_chain = LLMChain(
    llm=openai_api,  # You can also use the gemini_client here if necessary
    prompt=prompt_template
)

# Example interaction
user_message = "I am very anxious about this hackathon."
sentiment = "anxious"

# Get the response
response = llm_chain.run({
    "message": user_message,
    "sentiment": sentiment
})

print("Chatbot response:", response)
