from flask import Flask, request, jsonify
from gemini_api import GeminiClient  # Assuming this is the package for Gemini API
from transformers import pipeline

app = Flask(__name__)

# Initialize Gemini API and transformers pipeline
gemini_client = GeminiClient(api_key="your_gemini_api_key")  # Add your API key here
sentiment_analysis = pipeline("sentiment-analysis")

# Route for the chatbot interaction
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    
    # Get sentiment from user's message
    sentiment = sentiment_analysis(user_message)[0]
    
    # Call Gemini API for a response based on sentiment
    gemini_response = gemini_client.get_response(user_message, sentiment['label'])
    
    # Format the response
    response = {
        "user_message": user_message,
        "sentiment": sentiment['label'],
        "gemini_response": gemini_response['message']
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
