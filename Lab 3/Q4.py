import os
import time
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# --- RSA Section ---

def rsa_generate_keys():
    start = time.time()
    key = RSA.generate(2048)
    return key, time.time() - start

def rsa_encrypt_file(public_key, filename):
    aes_key = get_random_bytes(32)
    cipher_rsa = PKCS1_OAEP.new(public_key)
    enc_aes_key = cipher_rsa.encrypt(aes_key)

    iv = get_random_bytes(16)
    with open(filename, 'rb') as f:
        plaintext = f.read()

    start = time.time()
    cipher_aes = AES.new(aes_key, AES.MODE_CFB, iv=iv)
    ciphertext = cipher_aes.encrypt(plaintext)
    end = time.time()

    return enc_aes_key, iv, ciphertext, end - start

def rsa_decrypt_file(private_key, enc_aes_key, iv, ciphertext):
    cipher_rsa = PKCS1_OAEP.new(private_key)
    start = time.time()
    aes_key = cipher_rsa.decrypt(enc_aes_key)
    cipher_aes = AES.new(aes_key, AES.MODE_CFB, iv=iv)
    plaintext = cipher_aes.decrypt(ciphertext)
    end = time.time()
    return plaintext, end - start

# --- ECC Section (Hybrid ElGamal + AES-GCM) ---

def ecc_generate_keys():
    start = time.time()
    private_key = ec.generate_private_key(ec.SECP256R1())
    return private_key, time.time() - start

def ecc_encrypt_file(public_key, filename):
    ephemeral_key = ec.generate_private_key(ec.SECP256R1())
    shared_secret = ephemeral_key.exchange(ec.ECDH(), public_key)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'file-transfer',
    ).derive(shared_secret)

    iv = os.urandom(12)
    with open(filename, 'rb') as f:
        plaintext = f.read()

    start = time.time()
    encryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    end = time.time()

    ephemeral_pub_bytes = ephemeral_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )

    return ephemeral_pub_bytes, iv, encryptor.tag, ciphertext, end - start

def ecc_decrypt_file(private_key, ephemeral_pub_bytes, iv, tag, ciphertext):
    ephemeral_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ephemeral_pub_bytes)
    shared_secret = private_key.exchange(ec.ECDH(), ephemeral_pub)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'file-transfer',
    ).derive(shared_secret)

    start = time.time()
    decryptor = Cipher(algorithms.AES(derived_key), modes.GCM(iv, tag)).decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    end = time.time()

    return plaintext, end - start

# --- Main testing and reporting ---

def test_files(files):
    results = {"RSA": {}, "ECC": {}}

    for filename in files:
        print(f"\n--- Testing file: {filename} ---")

        # RSA
        rsa_key, rsa_key_time = rsa_generate_keys()
        enc_aes_key, iv, ciphertext, rsa_enc_time = rsa_encrypt_file(rsa_key.publickey(), filename)
        plaintext, rsa_dec_time = rsa_decrypt_file(rsa_key, enc_aes_key, iv, ciphertext)
        rsa_success = plaintext == open(filename, 'rb').read()

        results["RSA"][filename] = {
            "KeyGen": rsa_key_time,
            "Encrypt": rsa_enc_time,
            "Decrypt": rsa_dec_time,
            "Success": rsa_success,
            "CiphertextSize": len(ciphertext) + len(enc_aes_key) + len(iv)
        }

        print(f"RSA key gen time: {rsa_key_time:.3f}s")
        print(f"RSA encryption time: {rsa_enc_time:.3f}s")
        print(f"RSA decryption time: {rsa_dec_time:.3f}s")
        print(f"RSA successful decryption: {rsa_success}")

        # ECC
        ecc_priv_key, ecc_key_time = ecc_generate_keys()
        ephemeral_pub, iv, tag, ciphertext, ecc_enc_time = ecc_encrypt_file(ecc_priv_key.public_key(), filename)
        plaintext, ecc_dec_time = ecc_decrypt_file(ecc_priv_key, ephemeral_pub, iv, tag, ciphertext)
        ecc_success = plaintext == open(filename, 'rb').read()

        results["ECC"][filename] = {
            "KeyGen": ecc_key_time,
            "Encrypt": ecc_enc_time,
            "Decrypt": ecc_dec_time,
            "Success": ecc_success,
            "CiphertextSize": len(ciphertext) + len(ephemeral_pub) + len(iv) + len(tag)
        }

        print(f"ECC key gen time: {ecc_key_time:.3f}s")
        print(f"ECC encryption time: {ecc_enc_time:.3f}s")
        print(f"ECC decryption time: {ecc_dec_time:.3f}s")
        print(f"ECC successful decryption: {ecc_success}")

    return results

def print_summary(results, files):
    print("\n=== Summary of Results ===\n")
    for algo in ["RSA", "ECC"]:
        print(f"--- {algo} ---")
        for filename in files:
            res = results[algo][filename]
            print(f"File: {filename}")
            print(f"  KeyGen Time   : {res['KeyGen']:.3f}s")
            print(f"  Encrypt Time  : {res['Encrypt']:.3f}s")
            print(f"  Decrypt Time  : {res['Decrypt']:.3f}s")
            print(f"  Ciphertext Size (bytes): {res['CiphertextSize']}")
            print(f"  Decryption Success: {res['Success']}")
            print()

# --- Usage ---

if __name__ == "__main__":
    # Make sure test files exist or create dummy ones
    for fname, size in [("file1MB.bin", 1024*1024), ("file10MB.bin", 10*1024*1024)]:
        if not os.path.exists(fname):
            with open(fname, "wb") as f:
                f.write(os.urandom(size))
            print(f"Created test file {fname} of size {size} bytes.")

    files_to_test = ["file1MB.bin", "file10MB.bin"]

    results = test_files(files_to_test)
    print_summary(results, files_to_test)
