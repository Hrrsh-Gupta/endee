from pypdf import PdfReader
from google import genai
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, EMBEDDING_MODEL

class RAGSystem:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = LLM_MODEL
        self.embedding_model = EMBEDDING_MODEL

    def extract_text_from_pdf(self, file):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        return text.replace("\x00", "").strip()

    def chunk_text(self, text):
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def embed_texts(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_single(self, text):
        return self.embed_texts([text])[0]

    def generate_answer(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    def process_pdf(self, file):
        raw_text = self.extract_text_from_pdf(file)
        cleaned_text = self.clean_text(raw_text)

        if len(cleaned_text) < 500:
            raise ValueError("Document appears to be scanned or empty. OCR is needed.")

        return self.chunk_text(cleaned_text)