import voyageai
import numpy as np
import fitz
import time
import pickle
import openai
import re
import PyPDF2

vo = voyageai.Client(api_key="pa-gA94dVkc9_oN6GXJaheWdjCdrzwY06JoNOCDbqyBkqg") # When I ran this, I included my own key, but for debugging purposes I took it out

def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text("text")  # Correct the argument here
    return text


def chunk_text(text, chunk_size=100):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        current_chunk_word_count = len(' '.join(current_chunk).split())

        if current_chunk_word_count >= chunk_size:
            # Join the sentences in the current chunk and add to the chunks list
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def embed_texts(documents):
    for i, chunk in enumerate(documents):
        wc = len(chunk.split())
        print(f"Chunk {i+1} (Word count: {wc}):\n{chunk}\n")
    # print(documents)
    print(len(documents))

    # document_embeddings = []
    embedding = vo.embed(documents, model="voyage-3", input_type="document").embeddings # send the requests at once (10K token limit)
    time.sleep(5)
    return embedding

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def generate_questions(text, num_questions=10):
    prompt = f"Create {num_questions} quiz questions based on the following text:\n\n{text}"
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the GPT-4 Turbo model
        messages=[{"role": "user", "content": prompt}]
    )
    
    questions = completion.choices[0].message.content.strip().split('\n')
    return questions
