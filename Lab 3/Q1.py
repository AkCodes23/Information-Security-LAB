from Crypto.Util.number import bytes_to_long, long_to_bytes
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Generate RSA keys (for demonstration)
key = RSA.generate(2048)
public_key = key.publickey()

# Message to encrypt
message = "Asymmetric Encryption".encode()

# Encrypt with public key
cipher = PKCS1_OAEP.new(public_key)
ciphertext = cipher.encrypt(message)

print("Encrypted:", ciphertext.hex())

# Decrypt with private key
decipher = PKCS1_OAEP.new(key)
plaintext = decipher.decrypt(ciphertext)

print("Decrypted:", plaintext.decode())
