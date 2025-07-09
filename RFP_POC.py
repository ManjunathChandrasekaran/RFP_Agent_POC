import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import warnings


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


knowledge_base = {
    "What is your company's experience?": "Our company has 10+ years in AI and automation, delivering solutions for Fortune 500 clients.",
    "What is your pricing model?": "We offer flexible pricing based on project scope, typically subscription-based or per-use.",
    "How do you ensure data security?": "We comply with GDPR and ISO 27001, using AES-256 encryption and secure APIs."
}

past_proposals = {
    "What are your delivery timelines?": "Typical delivery within 8-12 weeks, depending on complexity.",
    "What is your support model?": "24/7 support with dedicated account managers and SLA-backed response times."
}

# Initialize Hugging Face models
generator = pipeline("text-generation", model="distilgpt2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Function to parse RFP text file
def parse_rfp(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        # Assume questions are separated by newlines or numbered
        questions = [q.strip() for q in content.split("\n") if q.strip()]
    return questions


# Function to find relevant context from knowledge base or past proposals
def retrieve_context(question):
    all_context = {**knowledge_base, **past_proposals}
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    context_keys = list(all_context.keys())
    context_embeddings = embedder.encode(context_keys, convert_to_tensor=True)

    # Compute similarity scores
    similarities = util.cos_sim(question_embedding, context_embeddings)[0]
    max_idx = similarities.argmax()
    if similarities[max_idx] > 0.5:  # Confidence threshold
        return all_context[context_keys[max_idx]]
    return None


# Function to generate response
def generate_response(question, context=None):
    if context:
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    response = generator(prompt, max_length=100, num_return_sequences=1, truncation=True)[0]["generated_text"]
    # Extract answer part
    answer = response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()
    return answer, 0.9 if context else 0.4  # Mock confidence score


# Function to flag for human-in-the-loop
def human_in_the_loop(question, response, confidence):
    if confidence < 0.5:
        print(f"Flagged for review: {question}")
        print(f"Generated response: {response}")
        human_input = input("Please provide or edit the response (or press Enter to keep): ")
        return human_input if human_input else response
    return response


# Main function to process RFP
def process_rfp(input_file, output_file):
    questions = parse_rfp(input_file)
    responses = []

    for question in questions:
        context = retrieve_context(question)
        response, confidence = generate_response(question, context)
        final_response = human_in_the_loop(question, response, confidence)
        responses.append(f"Question: {question}\nAnswer: {final_response}\n")

    # Write responses to output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.writelines(responses)


# Example usage
if __name__ == "__main__":
    input_file = r"C:\Users\xpman\PycharmProjects\RFP_POC\Input\rfp_input.txt"
    output_file = r"C:\Users\xpman\PycharmProjects\RFP_POC\Output\rfp_response.txt"

    # Create a sample input file for demo
    # with open(input_file, "w", encoding="utf-8") as f:
    #     f.write(
    #         "What is your company's experience?\nWhat is your pricing model?\nWhat is your unique value proposition?")

    process_rfp(input_file, output_file)
    print(f"Responses written to {output_file}")