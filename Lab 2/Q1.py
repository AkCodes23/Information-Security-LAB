from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

# Message and key
message = b"Confidential Data"
key = b"A1B2C3D4"  # DES key must be 8 bytes

# Create DES cipher in ECB mode
cipher = DES.new(key, DES.MODE_ECB)

# Encrypt
padded_text = pad(message, DES.block_size)
ciphertext = cipher.encrypt(padded_text)
print("Encrypted:", ciphertext.hex())

# Decrypt
decrypted_padded = cipher.decrypt(ciphertext)
decrypted = unpad(decrypted_padded, DES.block_size)
print("Decrypted:", decrypted.decode())
