import time
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def rsa_generate_keys():
    start = time.time()
    key = RSA.generate(2048)
    end = time.time()
    return key, end - start

def rsa_encrypt(public_key, plaintext):
    aes_key = get_random_bytes(32)
    cipher_rsa = PKCS1_OAEP.new(public_key)
    enc_aes_key = cipher_rsa.encrypt(aes_key)

    iv = get_random_bytes(16)
    cipher_aes = AES.new(aes_key, AES.MODE_CFB, iv=iv)
    start = time.time()
    ciphertext = cipher_aes.encrypt(plaintext)
    end = time.time()

    return enc_aes_key, iv, ciphertext, end - start

def rsa_decrypt(private_key, enc_aes_key, iv, ciphertext):
    cipher_rsa = PKCS1_OAEP.new(private_key)
    start = time.time()
    aes_key = cipher_rsa.decrypt(enc_aes_key)
    cipher_aes = AES.new(aes_key, AES.MODE_CFB, iv=iv)
    plaintext = cipher_aes.decrypt(ciphertext)
    end = time.time()
    return plaintext, end - start

def ecc_generate_keys():
    start = time.time()
    private_key = ec.generate_private_key(ec.SECP256R1())
    end = time.time()
    return private_key, end - start

def ecc_encrypt(public_key, plaintext):

    ephemeral_private = ec.generate_private_key(ec.SECP256R1())
    shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)


    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'elgamal encryption',
    ).derive(shared_secret)

    iv = os.urandom(12)
    encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv)).encryptor()
    start = time.time()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    end = time.time()

    ephemeral_public_bytes = ephemeral_private.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint
    )
    tag = encryptor.tag
    return ephemeral_public_bytes, iv, tag, ciphertext, end - start

def ecc_decrypt(private_key, ephemeral_public_bytes, iv, tag, ciphertext):
    ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ephemeral_public_bytes)
    shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'elgamal encryption',
    ).derive(shared_secret)

    start = time.time()
    decryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv, tag)).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    end = time.time()
    return plaintext, end - start


def generate_message(size_kb):
    return os.urandom(size_kb * 1024)

def run_test(size_kb):
    print(f"\nTesting message size: {size_kb} KB")

    message = generate_message(size_kb)

    print(f"RSA key gen time: {rsa_key_time:.4f} s")

    enc_aes_key, iv, ciphertext, rsa_enc_time = rsa_encrypt(rsa_key.publickey(), message)
    print(f"RSA encryption time: {rsa_enc_time:.4f} s")

    decrypted, rsa_dec_time = rsa_decrypt(rsa_key, enc_aes_key, iv, ciphertext)
    print(f"RSA decryption time: {rsa_dec_time:.4f} s")

    assert decrypted == message, "RSA decryption failed!"


    ecc_priv, ecc_key_time = ecc_generate_keys()
    print(f"ECC key gen time: {ecc_key_time:.4f} s")

    ephemeral_pub, iv, tag, ciphertext, ecc_enc_time = ecc_encrypt(ecc_priv.public_key(), message)
    print(f"ECC encryption time: {ecc_enc_time:.4f} s")

    decrypted, ecc_dec_time = ecc_decrypt(ecc_priv, ephemeral_pub, iv, tag, ciphertext)
    print(f"ECC decryption time: {ecc_dec_time:.4f} s")

    assert decrypted == message, "ECC decryption failed!"

if __name__ == "__main__":
    for size in [1, 10]:
        run_test(size)

