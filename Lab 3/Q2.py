from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

# Message
message = b"Secure Transactions"

# Generate private key for recipient (private)
private_key = ec.generate_private_key(ec.SECP256R1())

# Generate public key (sender would use this to encrypt)
public_key = private_key.public_key()

# --- Encrypt ---

# Ephemeral key (sender generates)
ephemeral_key = ec.generate_private_key(ec.SECP256R1())

# Shared secret from ephemeral private key and recipient's public key
shared_secret = ephemeral_key.exchange(ec.ECDH(), public_key)

# Derive symmetric key from shared secret
derived_key = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'handshake data',
).derive(shared_secret)

# Encrypt message with symmetric key (AES GCM)
iv = os.urandom(12)
encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv)).encryptor()
ciphertext = encryptor.update(message) + encryptor.finalize()

# Send ephemeral public key, ciphertext, iv, tag
ephemeral_public_bytes = ephemeral_key.public_key().public_bytes(
    encoding=serialization.Encoding.X962,
    format=serialization.PublicFormat.UncompressedPoint
)
tag = encryptor.tag

print("Encrypted ciphertext:", ciphertext.hex())

# --- Decrypt ---

# Load ephemeral public key
from cryptography.hazmat.primitives.serialization import load_der_public_key, Encoding, PublicFormat

ephemeral_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
    ec.SECP256R1(),
    ephemeral_public_bytes
)

# Recipient derives shared secret
shared_secret_dec = private_key.exchange(ec.ECDH(), ephemeral_public_key)

# Derive symmetric key same way
derived_key_dec = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'handshake data',
).derive(shared_secret_dec)

# Decrypt ciphertext
decryptor = Cipher(algorithms.AES(derived_key_dec), modes.GCM(iv, tag)).decryptor()
decrypted_message = decryptor.update(ciphertext) + decryptor.finalize()

print("Decrypted message:", decrypted_message.decode())
