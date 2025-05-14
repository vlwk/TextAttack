from vertexai.preview import tokenization

# Initialize tokenizer for Gemini 2.0 Flash
model_name = "gemini-1.5-flash-001"
tokenizer = tokenization.get_tokenizer_for_model(model_name)

# Base text
base_text = "restart"

# Define invisible characters
invisible_chars = {
    "U+200B (Zero Width Space)": "\u200B",
    "U+200C (Zero Width Non-Joiner)": "\u200C",
    "U+200D (Zero Width Joiner)": "\u200D",
}

# Token count for base text
base_count = tokenizer.count_tokens(base_text).total_tokens
print(f"Base text token count: {base_count}\n")

# Test each invisible character
for name, char in invisible_chars.items():
    # Insert invisible character between 're' and 'start'
    test_text = f"re{char}start"
    test_count = tokenizer.count_tokens(test_text).total_tokens
    print(f"{name} inserted:")
    print(f"Text: {repr(test_text)}")
    print(f"Token count: {test_count}")
    print(f"Difference from base: {test_count - base_count}\n")
