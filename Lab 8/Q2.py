from phe import paillier
from collections import defaultdict

# === 2a. Create a dataset ===
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

# === 2b. Paillier encryption/decryption ===
public_key, private_key = paillier.generate_paillier_keypair()

def encrypt_int(val):
    return public_key.encrypt(val)

def decrypt_int(enc_val):
    return private_key.decrypt(enc_val)

# === 2c. Create encrypted inverted index ===
# Map each document name to a unique integer ID
doc_id_map = {doc_name: idx + 1 for idx, doc_name in enumerate(documents.keys())}
reverse_doc_id_map = {v: k for k, v in doc_id_map.items()}

# Build inverted index: word → set of doc IDs
inverted_index = defaultdict(set)
for doc_name, content in documents.items():
    for word in content.lower().split():
        inverted_index[word].add(doc_id_map[doc_name])

# Encrypt the inverted index
encrypted_index = {}
for word, doc_ids in inverted_index.items():
    encrypted_doc_ids = [encrypt_int(doc_id) for doc_id in doc_ids]
    encrypted_index[word] = encrypted_doc_ids

# === 2d. Search function ===
def search(query):
    query = query.lower()
    if query in encrypted_index:
        enc_doc_ids = encrypted_index[query]
        doc_ids = [decrypt_int(enc_id) for enc_id in enc_doc_ids]
        print(f"\nSearch results for '{query}':")
        for doc_id in doc_ids:
            doc_name = reverse_doc_id_map[doc_id]
            print(f"{doc_name}: {documents[doc_name]}")
    else:
        print(f"\nNo results found for '{query}'")

# === Example usage ===
if __name__ == "__main__":
    queries = ["fox", "music", "loyal", "quick", "animals"]
    for q in queries:
        search(q)
