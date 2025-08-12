from Crypto.Util.number import bytes_to_long, long_to_bytes
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP


key = RSA.generate(2048)
public_key = key.publickey()


message = "Asymmetric Encryption".encode()


cipher = PKCS1_OAEP.new(public_key)
ciphertext = cipher.encrypt(message)

print("Encrypted:", ciphertext.hex())


decipher = PKCS1_OAEP.new(key)
plaintext = decipher.decrypt(ciphertext)

print("Decrypted:", plaintext.decode())

