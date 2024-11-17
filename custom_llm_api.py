from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Create a Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


# Define the LLM API wrapper function using GPT-2 model
def my_llm_api(
    prompt: str,
    instructions: str,
    temperature: float = 0.1,
    max_length: int = 1024,
    **kwargs,
) -> str:
    """Custom LLM API wrapper using GPT-2 model."""
    try:
        # Tokenize input prompt and instructions
        input_prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_instruction_ids = tokenizer.encode(instructions, return_tensors="pt")

        # Concatenate input prompt and instructions
        input_ids = torch.cat([input_prompt_ids, input_instruction_ids], dim=-1)

        # Generate text based on the concatenated input using GPT-2 model
        output = model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            **kwargs,
        )

        # Decode the generated text
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text  # Return the generated text
    except Exception as e:
        # Handle errors gracefully
        print(f"An error occurred while generating text: {str(e)}")
        return ""  # Return an empty string in case of an error


# Define a route for the LLM API
@cross_origin()
@app.route("/generate_text", methods=["POST"])
def generate_text():
    # Parse JSON request
    data = request.json
    prompt = data.get("prompt", "")
    instructions = data.get("instructions", "")
    temperature = data.get("temperature", 0.1)

    print(prompt, instructions)

    print("Generating text...")

    # Call the LLM API wrapper function
    generated_text = my_llm_api(
        prompt=prompt, instructions=instructions, temperature=temperature
    )

    # Return the generated text as JSON response
    return jsonify({"generated_text": generated_text})


if __name__ == "__main__":
    # Run the Flask app
    app.run(port=8001, debug=True)
