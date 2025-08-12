import os
import time
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def generate_ecc_keypair():
    priv_key = ec.generate_private_key(ec.SECP256R1())
    pub_key = priv_key.public_key()
    return priv_key, pub_key

def ecc_elgamal_encrypt(pub_key, plaintext_bytes):
    ephemeral_priv = ec.generate_private_key(ec.SECP256R1())
    ephemeral_pub = ephemeral_priv.public_key()

    shared_secret = ephemeral_priv.exchange(ec.ECDH(), pub_key)

    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'elgamal-encryption',
    ).derive(shared_secret)

    iv = os.urandom(12)
    encryptor = Cipher(algorithms.AES(aes_key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(plaintext_bytes) + encryptor.finalize()


    ephemeral_pub_bytes = ephemeral_pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )

    return ephemeral_pub_bytes, iv, encryptor.tag, ciphertext

def ecc_elgamal_decrypt(priv_key, ephemeral_pub_bytes, iv, tag, ciphertext):

    ephemeral_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(),
        ephemeral_pub_bytes
    )

    shared_secret = priv_key.exchange(ec.ECDH(), ephemeral_pub)

    aes_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'elgamal-encryption',
    ).derive(shared_secret)

    decryptor = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag)).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    return plaintext

def generate_patient_data(size_bytes):
    return os.urandom(size_bytes)

data_sizes = [128, 1024, 10*1024]  # bytes: 128B, 1KB, 10KB

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

