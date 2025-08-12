from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


message = b"Top Secret Data"
key = bytes.fromhex("FEDCBA9876543210FEDCBA9876543210") 


cipher = AES.new(key, AES.MODE_ECB)

padded = pad(message, AES.block_size)
ciphertext = cipher.encrypt(padded)
print("Encrypted:", ciphertext.hex())

decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
print("Decrypted:", decrypted.decode())

