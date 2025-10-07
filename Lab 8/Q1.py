from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import hashlib
from collections import defaultdict


# === 1a. Create a dataset ===
documents = {
    "doc1": "the quick brown fox jumps over the lazy dog",
    "doc2": "never jump over the lazy dog quickly",
    "doc3": "bright sun shines over the hills",
    "doc4": "the fox is clever and quick",
    "doc5": "dogs are loyal and friendly animals",
    "doc6": "the hills are alive with the sound of music",
    "doc7": "quick thinking leads to smart decisions",
    "doc8": "music soothes the soul and calms the mind",
    "doc9": "the clever dog outsmarted the fox",
    "doc10": "friendly animals make great companions"
}

# === 1b. AES Encryption/Decryption ===
key = hashlib.sha256(b'secret_key').digest()

def pad(text):
    return text + (16 - len(text) % 16) * chr(16 - len(text) % 16)

def unpad(text):
    return text[:-ord(text[-1])]

def encrypt(text):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_text = pad(text)
    encrypted = cipher.encrypt(padded_text.encode())
    return base64.b64encode(encrypted).decode()

def decrypt(enc_text):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = cipher.decrypt(base64.b64decode(enc_text))
    return unpad(decrypted.decode())

# === 1c. Create and encrypt inverted index ===
inverted_index = defaultdict(set)
for doc_id, content in documents.items():
    for word in content.lower().split():
        inverted_index[word].add(doc_id)

encrypted_index = {}
for word, doc_ids in inverted_index.items():
    encrypted_word = encrypt(word)
    encrypted_doc_ids = [encrypt(doc_id) for doc_id in doc_ids]
    encrypted_index[encrypted_word] = encrypted_doc_ids

# === 1d. Search function ===
def search(query):
    encrypted_query = encrypt(query.lower())
    if encrypted_query in encrypted_index:
        encrypted_doc_ids = encrypted_index[encrypted_query]
        doc_ids = [decrypt(doc_id) for doc_id in encrypted_doc_ids]
        print(f"\nSearch results for '{query}':")
        for doc_id in doc_ids:
            print(f"{doc_id}: {documents[doc_id]}")
    else:
        print(f"\nNo results found for '{query}'")

# === Example usage ===
if __name__ == "__main__":
    queries = ["fox", "music", "loyal", "quick", "animals"]
    for q in queries:
        search(q)

