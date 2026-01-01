#utils
import re # regex operations

def sanitize_text(s): # sanitize text by removing extra spaces and newlines
    return " ".join(s.split())

def chunk_text(s, n=2000): # chunk text into n-word pieces
    words = s.split()
    chunks = []
    for i in range(0, len(words), n):
        chunks.append(" ".join(words[i:i+n]))
    return chunks