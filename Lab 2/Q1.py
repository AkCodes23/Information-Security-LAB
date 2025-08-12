from Crypto.Cipher import DES
from Crypto.Util.Padding import pad, unpad

message = b"Confidential Data"
key = b"A1B2C3D4"  # DES key must be 8 bytes


cipher = DES.new(key, DES.MODE_ECB)


padded_text = pad(message, DES.block_size)
ciphertext = cipher.encrypt(padded_text)
print("Encrypted:", ciphertext.hex())


decrypted_padded = cipher.decrypt(ciphertext)
decrypted = unpad(decrypted_padded, DES.block_size)
print("Decrypted:", decrypted.decode())
