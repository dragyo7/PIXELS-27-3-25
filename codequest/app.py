from flask import Flask, request, render_template
from transformers import pipeline
import re
import gc
import torch

app = Flask(__name__)

# Global generator variable
generator = None

def load_model():
    global generator
    if generator is None:
        generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    return generator

def unload_model():
    global generator
    if generator is not None:
        del generator
        torch.cuda.empty_cache()  # Clear GPU memory (if used)
        gc.collect()  # Force garbage collection
        generator = None

# Personalization function
def personalize(text, audience):
    if "millennials" in audience.lower():
        text = text.replace("technology", "tech").replace("people", "peeps")
    elif "professionals" in audience.lower():
        text = text.replace("tech", "technology").replace("peeps", "professionals")
    return text

# Clean up function
def clean_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    complete_sentences = [s for s in sentences if s.endswith(('.', '!', '?'))]
    seen = set()
    unique_sentences = [s for s in complete_sentences if not (s in seen or seen.add(s))]
    return " ".join(unique_sentences)

@app.route("/", methods=["GET", "POST"])
def generate_content():
    if request.method == "POST":
        topic = request.form["topic"]
        audience = request.form["audience"]
        tone = request.form["tone"]
        
        # Load GPT-Neo model
        gen = load_model()
        
        # Structured prompt
        prompt = f"Write a {tone} 400-word article about {topic} for {audience}. Start with an introduction, followed by key points, and end with a conclusion."
        
        # Generate content
        output = gen(
            prompt,
            max_length=800,  # ~400-500 words, within 2048-token limit
            num_return_sequences=1,
            temperature=0.7,
            top_k=40,
            truncation=True
        )[0]["generated_text"]
        
        # Clean and personalize
        raw_content = output[len(prompt):].strip()
        cleaned_content = clean_text(raw_content)
        personalized_content = personalize(cleaned_content, audience)
        
        return render_template("result.html", content=personalized_content)
    return render_template("index.html")

@app.route("/clear", methods=["GET"])
def clear_memory():
    unload_model()
    return "Memory cleared! <a href='/'>Back to Generator</a>"

if __name__ == "__main__":
    app.run(debug=True)