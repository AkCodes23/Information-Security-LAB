from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Message and key
message = b"Sensitive Information"
key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")  # 16-byte key

# Create AES cipher in ECB mode
cipher = AES.new(key, AES.MODE_ECB)

# Encrypt
padded_text = pad(message, AES.block_size)
ciphertext = cipher.encrypt(padded_text)
print("Encrypted:", ciphertext.hex())

# Decrypt
decrypted_padded = cipher.decrypt(ciphertext)
decrypted = unpad(decrypted_padded, AES.block_size)
print("Decrypted:", decrypted.decode())
