import tiktoken
import time
import threading

# Values from https://platform.openai.com/docs/models/gpt-3-5
MODELS_2_TOKEN_LIMITS = {
    "gpt-35-turbo": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-35-turbo-16k": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 128000
}

AOAI_2_OAI = {
    "gpt-35-turbo": "gpt-3.5-turbo",
    "gpt-35-turbo-16k": "gpt-3.5-turbo-16k",
    "gpt-4o": "gpt-4o"
}

user_token_usage = {}
MAX_TOKENS_PER_DAY = 1000

def log_token_usage(user_id, tokens_used):
    if user_id in user_token_usage:
        user_token_usage[user_id] += tokens_used
    else:
        user_token_usage[user_id] = tokens_used

def check_token_limit(user_id):
    if user_id in user_token_usage:
        return user_token_usage[user_id] < MAX_TOKENS_PER_DAY
    return True

def process_request(user_id, tokens_needed):
    if check_token_limit(user_id):
        log_token_usage(user_id, tokens_needed)
        return "Request processed"
    else:
        return "Token limit exceeded. Please try again later."

def reset_token_usage():
    global user_token_usage
    while True:
        time.sleep(86400)  # Sleep for 24 hours
        user_token_usage = {}

reset_thread = threading.Thread(target=reset_token_usage)
reset_thread.start()

def get_token_limit(model_id: str) -> int:
    if model_id not in MODELS_2_TOKEN_LIMITS:
        raise ValueError("Expected model gpt-35-turbo and above. Got: " + model_id)
    return MODELS_2_TOKEN_LIMITS.get(model_id)

def num_tokens_from_messages(message: dict[str, str], model: str) -> int:
    """
    Calculate the number of tokens required to encode a message.
    Args:
        message (dict): The message to encode, represented as a dictionary.
        model (str): The name of the model to use for encoding.
    Returns:
        int: The total number of tokens required to encode the message.
    Example:
        message = {'role': 'user', 'content': 'Hello, how are you?'}
        model = 'gpt-3.5-turbo'
        num_tokens_from_messages(message, model)
        output: 11
    """
    encoding = tiktoken.encoding_for_model(get_oai_chatmodel_tiktok(model))
    num_tokens = 2  # For "role" and "content" keys
    for key, value in message.items():
        num_tokens += len(encoding.encode(value))
    return num_tokens

def get_oai_chatmodel_tiktok(aoaimodel: str) -> str:
    message = "Expected Azure OpenAI ChatGPT model name"
    if aoaimodel == "" or aoaimodel is None:
        raise ValueError(message)
    if aoaimodel not in AOAI_2_OAI and aoaimodel not in MODELS_2_TOKEN_LIMITS:
        raise ValueError(message)
    return AOAI_2_OAI.get(aoaimodel) or aoaimodel

def handle_user_request(user_id, user_input):
    tokens_needed = num_tokens_from_messages(user_input, "gpt-4")  # Adjust model as needed
    response = process_request(user_id, tokens_needed)
    if response == "Token limit exceeded. Please try again later.":
        return response
    else:
        # Proceed with processing the request
        result = process_user_input(user_input)
        return result