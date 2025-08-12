from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF")
message = b"Encryption Strength"

cipher = AES.new(key, AES.MODE_ECB)

ciphertext = cipher.encrypt(pad(message, AES.block_size))
print("AES-256 Ciphertext:", ciphertext.hex())

decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
print("AES-256 Decrypted:", decrypted.decode())

