from Crypto.Cipher import AES
from Crypto.Util import Counter

# AES-256 key and nonce
key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF")
nonce = bytes.fromhex("0000000000000000")  # 8 bytes
message = b"Cryptography Lab Exercise"

# Create counter object for CTR mode
ctr = Counter.new(64, prefix=nonce, initial_value=0)

# Encrypt
cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
ciphertext = cipher.encrypt(message)
print("AES-256 CTR Ciphertext:", ciphertext.hex())

# Decrypt (recreate counter)
ctr = Counter.new(64, prefix=nonce, initial_value=0)
decipher = AES.new(key, AES.MODE_CTR, counter=ctr)
decrypted = decipher.decrypt(ciphertext)
print("AES-256 CTR Decrypted:", decrypted.decode())
