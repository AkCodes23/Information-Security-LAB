from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# AES-256 key and message
key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF")
message = b"Encryption Strength"

# Create AES cipher in ECB mode
cipher = AES.new(key, AES.MODE_ECB)

# Encrypt
ciphertext = cipher.encrypt(pad(message, AES.block_size))
print("AES-256 Ciphertext:", ciphertext.hex())

# Decrypt
decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
print("AES-256 Decrypted:", decrypted.decode())
