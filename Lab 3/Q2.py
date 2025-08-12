from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os


message = b"Secure Transactions"

private_key = ec.generate_private_key(ec.SECP256R1())


public_key = private_key.public_key()

ephemeral_key = ec.generate_private_key(ec.SECP256R1())


shared_secret = ephemeral_key.exchange(ec.ECDH(), public_key)

derived_key = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'handshake data',
).derive(shared_secret)


iv = os.urandom(12)
encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv)).encryptor()
ciphertext = encryptor.update(message) + encryptor.finalize()


ephemeral_public_bytes = ephemeral_key.public_key().public_bytes(
    encoding=serialization.Encoding.X962,
    format=serialization.PublicFormat.UncompressedPoint
)
tag = encryptor.tag

print("Encrypted ciphertext:", ciphertext.hex())


from cryptography.hazmat.primitives.serialization import load_der_public_key, Encoding, PublicFormat

ephemeral_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
    ec.SECP256R1(),
    ephemeral_public_bytes
)


shared_secret_dec = private_key.exchange(ec.ECDH(), ephemeral_public_key)


derived_key_dec = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'handshake data',
).derive(shared_secret_dec)


decryptor = Cipher(algorithms.AES(derived_key_dec), modes.GCM(iv, tag)).decryptor()
decrypted_message = decryptor.update(ciphertext) + decryptor.finalize()

print("Decrypted message:", decrypted_message.decode())

