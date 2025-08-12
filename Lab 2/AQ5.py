from Crypto.Cipher import AES
from Crypto.Util import Counter

key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF")
nonce = bytes.fromhex("0000000000000000") 
message = b"Cryptography Lab Exercise"

ctr = Counter.new(64, prefix=nonce, initial_value=0)

cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
ciphertext = cipher.encrypt(message)
print("AES-256 CTR Ciphertext:", ciphertext.hex())

ctr = Counter.new(64, prefix=nonce, initial_value=0)
decipher = AES.new(key, AES.MODE_CTR, counter=ctr)
decrypted = decipher.decrypt(ciphertext)
print("AES-256 CTR Decrypted:", decrypted.decode())

