from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

# Load the model and tokenizer
mname = "facebook/blenderbot-3B"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = AutoTokenizer.from_pretrained(mname)

# Initialize conversation history
conversation_history = []

# Main loop to chat with the user
while True:
    # Get user input
    user_input = input("Human: ")
    
    # Check if the user wants to quit
    if user_input.lower() == 'quit':
        break
    
    # Add user input to conversation history
    conversation_history.append(user_input)
    
    # Prepare model inputs
    model_input = ' '.join(f'<s> {turn} </s>' for turn in conversation_history)
    inputs = tokenizer([model_input], return_tensors="pt")
    
    # Generate a response
    reply_ids = model.generate(**inputs, max_length=1000)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    
    # Add model response to conversation history
    conversation_history.append(response)
    
    print("Bot: ", response)
