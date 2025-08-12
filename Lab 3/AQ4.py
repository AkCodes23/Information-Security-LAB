import os
import time
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# ECC ElGamal-like encryption: encrypt a symmetric key

def generate_ecc_keypair():
    priv_key = ec.generate_private_key(ec.SECP256R1())
    pub_key = priv_key.public_key()
    return priv_key, pub_key

def ecc_elgamal_encrypt(pub_key, plaintext_bytes):
    # 1. Generate ephemeral key pair
    ephemeral_priv = ec.generate_private_key(ec.SECP256R1())
    ephemeral_pub = ephemeral_priv.public_key()

    # 2. Compute shared secret
    shared_secret = ephemeral_priv.exchange(ec.ECDH(), pub_key)

    # 3. Derive AES key from shared secret
    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'elgamal-encryption',
    ).derive(shared_secret)

    # 4. Encrypt plaintext symmetrically with AES-GCM
    iv = os.urandom(12)
    encryptor = Cipher(algorithms.AES(aes_key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(plaintext_bytes) + encryptor.finalize()

    # 5. Export ephemeral public key bytes (to send along)
    ephemeral_pub_bytes = ephemeral_pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )

    return ephemeral_pub_bytes, iv, encryptor.tag, ciphertext

def ecc_elgamal_decrypt(priv_key, ephemeral_pub_bytes, iv, tag, ciphertext):
    # Load ephemeral public key
    ephemeral_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(),
        ephemeral_pub_bytes
    )

    # Derive shared secret
    shared_secret = priv_key.exchange(ec.ECDH(), ephemeral_pub)

    # Derive AES key same way
    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'elgamal-encryption',
    ).derive(shared_secret)

    # Decrypt with AES-GCM
    decryptor = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag)).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    return plaintext

# Helper to generate random patient data of given size in bytes
def generate_patient_data(size_bytes):
    return os.urandom(size_bytes)

# Test with varying data sizes
data_sizes = [128, 1024, 10*1024]  # bytes: 128B, 1KB, 10KB

# Generate keys once
priv_key, pub_key = generate_ecc_keypair()

for size in data_sizes:
    print(f"\n--- Testing data size: {size} bytes ---")
    patient_data = generate_patient_data(size)

    start_enc = time.time()
    ephemeral_pub_bytes, iv, tag, ciphertext = ecc_elgamal_encrypt(pub_key, patient_data)
    end_enc = time.time()

    start_dec = time.time()
    decrypted = ecc_elgamal_decrypt(priv_key, ephemeral_pub_bytes, iv, tag, ciphertext)
    end_dec = time.time()

    print(f"Encryption time: {(end_enc - start_enc)*1000:.2f} ms")
    print(f"Decryption time: {(end_dec - start_dec)*1000:.2f} ms")
    print(f"Decryption successful: {decrypted == patient_data}")
