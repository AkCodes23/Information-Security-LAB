from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


message = b"Sensitive Information"
key = bytes.fromhex("0123456789ABCDEF0123456789ABCDEF")  # 16-byte key


cipher = AES.new(key, AES.MODE_ECB)


padded_text = pad(message, AES.block_size)
ciphertext = cipher.encrypt(padded_text)
print("Encrypted:", ciphertext.hex())

decrypted_padded = cipher.decrypt(ciphertext)
decrypted = unpad(decrypted_padded, AES.block_size)
print("Decrypted:", decrypted.decode())

