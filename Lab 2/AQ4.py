from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

key = b"A1B2C3D4"
iv = b"12345678"
message = b"Secure Communication"

cipher = DES.new(key, DES.MODE_CBC, iv)

ciphertext = cipher.encrypt(pad(message, DES.block_size))
print("DES CBC Ciphertext:", ciphertext.hex())

decipher = DES.new(key, DES.MODE_CBC, iv)
decrypted = unpad(decipher.decrypt(ciphertext), DES.block_size)
print("DES CBC Decrypted:", decrypted.decode())

