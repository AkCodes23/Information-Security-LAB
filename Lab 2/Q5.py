from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# Message and key
message = b"Top Secret Data"
key = bytes.fromhex("FEDCBA9876543210FEDCBA9876543210")  # 24 bytes for AES-192

# Create AES cipher
cipher = AES.new(key, AES.MODE_ECB)

# Encrypt
padded = pad(message, AES.block_size)
ciphertext = cipher.encrypt(padded)
print("Encrypted:", ciphertext.hex())

# Decrypt
decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
print("Decrypted:", decrypted.decode())
