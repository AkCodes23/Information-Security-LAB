from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

message = b"Secure Transactions"

# Generate ECC key pair (secp256r1)
private_key = ec.generate_private_key(ec.SECP256R1())
public_key = private_key.public_key()

# Encrypt
ephemeral_private = ec.generate_private_key(ec.SECP256R1())
shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)
derived_key = HKDF(hashes.SHA256(), 32, None, b'ecc encryption').derive(shared_secret)
iv = os.urandom(12)
encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv)).encryptor()
ciphertext = encryptor.update(message) + encryptor.finalize()
tag = encryptor.tag
ephemeral_public_bytes = ephemeral_private.public_key().public_bytes(
    serialization.Encoding.X962,
    serialization.PublicFormat.UncompressedPoint)

# Decrypt
ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ephemeral_public_bytes)
shared_secret_dec = private_key.exchange(ec.ECDH(), ephemeral_public)
derived_key_dec = HKDF(hashes.SHA256(), 32, None, b'ecc encryption').derive(shared_secret_dec)
decryptor = Cipher(algorithms.AES(derived_key_dec), modes.GCM(iv, tag)).decryptor()
decrypted = decryptor.update(ciphertext) + decryptor.finalize()

print(f"Original message: {message.decode()}")
print(f"Decrypted message: {decrypted.decode()}")
