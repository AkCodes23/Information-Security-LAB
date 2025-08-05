from Crypto.Cipher import DES3
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Message
message = b"Classified Text"

# Generate a strong 3DES key
while True:
    try:
        key = DES3.adjust_key_parity(get_random_bytes(24))
        cipher = DES3.new(key, DES3.MODE_ECB)
        break
    except ValueError:
        continue  # Retry if key degenerates

# Encrypt
padded_text = pad(message, DES3.block_size)
ciphertext = cipher.encrypt(padded_text)
print("Encrypted:", ciphertext.hex())

# Decrypt
decrypted_padded = cipher.decrypt(ciphertext)
decrypted = unpad(decrypted_padded, DES3.block_size)
print("Decrypted:", decrypted.decode())
