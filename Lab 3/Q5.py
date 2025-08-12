import time
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

def rsa_generate_keys():
    start = time.time()
    key = RSA.generate(2048)
    return key, time.time() - start

def rsa_encrypt_file(pub_key, filename):
    aes_key = get_random_bytes(32)
    enc_aes_key = PKCS1_OAEP.new(pub_key).encrypt(aes_key)
    iv = get_random_bytes(16)
    with open(filename, 'rb') as f:
        plaintext = f.read()
    start = time.time()
    ciphertext = AES.new(aes_key, AES.MODE_CFB, iv=iv).encrypt(plaintext)
    return enc_aes_key, iv, ciphertext, time.time() - start

def rsa_decrypt_file(priv_key, enc_aes_key, iv, ciphertext):
    start = time.time()
    aes_key = PKCS1_OAEP.new(priv_key).decrypt(enc_aes_key)
    plaintext = AES.new(aes_key, AES.MODE_CFB, iv=iv).decrypt(ciphertext)
    return plaintext, time.time() - start

def ecc_generate_keys():
    start = time.time()
    priv_key = ec.generate_private_key(ec.SECP256R1())
    return priv_key, time.time() - start

def ecc_encrypt_file(pub_key, filename):
    ephemeral_key = ec.generate_private_key(ec.SECP256R1())
    shared_secret = ephemeral_key.exchange(ec.ECDH(), pub_key)
    derived_key = HKDF(hashes.SHA256(), 32, None, b'handshake data').derive(shared_secret)
    iv = os.urandom(12)
    with open(filename, 'rb') as f:
        plaintext = f.read()
    start = time.time()
    encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    ephemeral_pub_bytes = ephemeral_key.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint
    )
    return ephemeral_pub_bytes, iv, encryptor.tag, ciphertext, time.time() - start

def ecc_decrypt_file(priv_key, ephemeral_pub_bytes, iv, tag, ciphertext):
    ephemeral_pub_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ephemeral_pub_bytes)
    shared_secret = priv_key.exchange(ec.ECDH(), ephemeral_pub_key)
    derived_key = HKDF(hashes.SHA256(), 32, None, b'handshake data').derive(shared_secret)
    start = time.time()
    decryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv, tag)).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext, time.time() - start

files = ["file1MB.bin", "file10MB.bin"]

for file in files:
    print(f"--- Testing {file} ---")

    rsa_key, rsa_key_time = rsa_generate_keys()
    print(f"RSA key gen time: {rsa_key_time:.4f}s")

    enc_aes_key, iv, ciphertext, rsa_enc_time = rsa_encrypt_file(rsa_key.publickey(), file)
    print(f"RSA encryption time: {rsa_enc_time:.4f}s")

    plaintext, rsa_dec_time = rsa_decrypt_file(rsa_key, enc_aes_key, iv, ciphertext)
    print(f"RSA decryption time: {rsa_dec_time:.4f}s")

    ecc_priv_key, ecc_key_time = ecc_generate_keys()
    print(f"ECC key gen time: {ecc_key_time:.4f}s")

    ephemeral_pub, iv, tag, ciphertext, ecc_enc_time = ecc_encrypt_file(ecc_priv_key.public_key(), file)
    print(f"ECC encryption time: {ecc_enc_time:.4f}s")

    plaintext, ecc_dec_time = ecc_decrypt_file(ecc_priv_key, ephemeral_pub, iv, tag, ciphertext)
    print(f"ECC decryption time: {ecc_dec_time:.4f}s\n")
