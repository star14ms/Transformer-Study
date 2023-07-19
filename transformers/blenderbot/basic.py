from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the model and tokenizer
model_name = 'facebook/blenderbot-3B'
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

# Function to generate a response
def generate_response(user_input):
    # Encode the user's input and end of string token
    inputs = tokenizer([user_input], return_tensors='pt')
    
    # Generate a response
    reply_ids = model.generate(**inputs, max_length=60)
    
    # Decode the response
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
    
    return response[0]

# Main loop to chat with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print("Bot: ", response)
