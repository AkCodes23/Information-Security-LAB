def rsa_encrypt_chunk(m_int, e, n):
    return pow(m_int, e, n)

def rsa_decrypt_chunk(c, d, n):
    return pow(c, d, n)

def split_message(msg_bytes, chunk_size):
    return [msg_bytes[i:i+chunk_size] for i in range(0, len(msg_bytes), chunk_size)]

def encrypt_message(msg, e, n):
    msg_bytes = msg.encode('utf-8')
    # Max chunk size in bytes so int(chunk) < n
    # Since n=323, max int chunk is 322 -> max chunk bytes = 1 (because 2 bytes = 65535 > 323)
    chunk_size = 1
    chunks = split_message(msg_bytes, chunk_size)
    encrypted_chunks = []
    for chunk in chunks:
        m_int = int.from_bytes(chunk, 'big')
        c = rsa_encrypt_chunk(m_int, e, n)
        encrypted_chunks.append(c)
    return encrypted_chunks

def decrypt_message(encrypted_chunks, d, n):
    decrypted_bytes = bytearray()
    for c in encrypted_chunks:
        m_int = rsa_decrypt_chunk(c, d, n)
        # convert decrypted int back to bytes (chunk size 1)
        decrypted_bytes.extend(m_int.to_bytes(1, 'big'))
    return decrypted_bytes.decode('utf-8')

# RSA keys and message
n = 323
e = 5
d = 173

message = "Cryptographic Protocols"

print(f"Original message: {message}")

encrypted_chunks = encrypt_message(message, e, n)
print(f"Encrypted chunks: {encrypted_chunks}")

decrypted_message = decrypt_message(encrypted_chunks, d, n)
print(f"Decrypted message: {decrypted_message}")
