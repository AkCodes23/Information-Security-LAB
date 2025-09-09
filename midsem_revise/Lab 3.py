# Lab 3: RSA, ECC (ECIES-style), ElGamal, Diffie-Hellman, file transfer perf
# Requires: pip install pycryptodome cryptography
# Run: python lab3_asymmetric.py

import os, time, secrets
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.PublicKey import RSA, ElGamal
from Crypto.Random import get_random_bytes, random
from Crypto.Util.number import getPrime, inverse
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization, kdf
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# RSA
def rsa_keypair(bits=2048):
    key = RSA.generate(bits)
    return key, key.publickey()

def rsa_encrypt(pub, data: bytes):
    cipher = PKCS1_OAEP.new(pub)
    return cipher.encrypt(data)

def rsa_decrypt(priv, ct: bytes):
    cipher = PKCS1_OAEP.new(priv)
    return cipher.decrypt(ct)

# Elliptic-curve ECIES-like (ECDH + AES-GCM)
def ec_keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    return priv, pub

def ecies_encrypt(pub, plaintext: bytes):
    eph_priv = ec.generate_private_key(ec.SECP256R1())
    shared = eph_priv.exchange(ec.ECDH(), pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ecies").derive(shared)
    iv = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    eph_pub_bytes = eph_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return eph_pub_bytes, iv, tag, ct

def ecies_decrypt(priv, package):
    eph_pub_bytes, iv, tag, ct = package
    eph_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), eph_pub_bytes)
    shared = priv.exchange(ec.ECDH(), eph_pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ecies").derive(shared)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    pt = cipher.decrypt_and_verify(ct, tag)
    return pt

# ElGamal over Z_p (PyCryptodome)
def elgamal_keypair(bits=256):
    p = getPrime(bits)
    g = random.randrange(2, p-1)
    key = ElGamal.generate(bits, get_random_bytes)
    # PyCryptodome ElGamal.generate returns valid key; keep it simple
    return key, key.publickey()

def elgamal_encrypt(pub, m: int):
    # ElGamal here expects integers mod p; we encode bytes -> int
    p = pub.p
    k = random.StrongRandom().randint(1, p-2)
    # Use message as integer < p
    return ElGamal.construct((pub.p, pub.g, pub.y)).encrypt(m, k)

def elgamal_decrypt(priv, ct):
    return priv.decrypt(ct)

# Diffie–Hellman (mod p)
def dh_keypair(p=None, g=2, bits=2048):
    # For demo: generate a prime p, or use a known safe prime for speed (omitted)
    if p is None:
        p = getPrime(bits)
    a = secrets.randbelow(p-2) + 2
    A = pow(g, a, p)
    return (p, g, a, A)

def dh_shared(p, g, a, B):
    return pow(B, a, p)

# File encryption helpers (RSA wrap AES)
def rsa_file_encrypt(pub, data: bytes):
    sym = get_random_bytes(32)
    iv = get_random_bytes(12)
    ct = AES.new(sym, AES.MODE_GCM, nonce=iv)
    cdata, tag = ct.encrypt_and_digest(data)
    wrapped = rsa_encrypt(pub, sym)
    return wrapped, iv, tag, cdata

def rsa_file_decrypt(priv, package):
    wrapped, iv, tag, cdata = package
    sym = rsa_decrypt(priv, wrapped)
    pt = AES.new(sym, AES.MODE_GCM, nonce=iv).decrypt_and_verify(cdata, tag)
    return pt

if __name__ == "__main__":
    # 1) RSA message
    msg = b"Asymmetric Encryption"
    rsa_priv, rsa_pub = rsa_keypair()
    c = rsa_encrypt(rsa_pub, msg)
    print("RSA:", rsa_decrypt(rsa_priv, c).decode())

    # 2) ECC ECIES-like
    ecc_priv, ecc_pub = ec_keypair()
    pkg = ecies_encrypt(ecc_pub, b"Secure Transactions")
    print("ECC:", ecies_decrypt(ecc_priv, pkg).decode())

    # 3) ElGamal message as integer (encode bytes to int < p)
    el_priv, el_pub = elgamal_keypair(256)
    m_int = int.from_bytes(b"Confidential Data", "big") % el_pub.p
    ct = elgamal_encrypt(el_pub, m_int)
    dec_int = elgamal_decrypt(el_priv, ct)
    print("ElGamal:", dec_int.to_bytes((dec_int.bit_length()+7)//8, "big"))

    # 4) Secure file transfer performance: RSA-2048 vs ECC-ECIES (1MB/10MB)
    for size in [1_000_000, 10_000_000]:
        data = os.urandom(size)
        # RSA wrap
        t0 = time.perf_counter(); pkg = rsa_file_encrypt(rsa_pub, data); t1 = time.perf_counter()
        _ = rsa_file_decrypt(rsa_priv, pkg); t2 = time.perf_counter()
        print(f"RSA-2048 file {size/1e6:.0f}MB enc:{t1-t0:.3f}s dec:{t2-t1:.3f}s")

        # ECC ECIES
        t0 = time.perf_counter(); pkg = ecies_encrypt(ecc_pub, data); t1 = time.perf_counter()
        _ = ecies_decrypt(ecc_priv, pkg); t2 = time.perf_counter()
        print(f"ECC-P256 file {size/1e6:.0f}MB enc:{t1-t0:.3f}s dec:{t2-t1:.3f}s")

    # 5) Diffie–Hellman key exchange timing
    p,g,a,A = dh_keypair(bits=512)  # smaller for speed in demo
    _,_,b,B = dh_keypair(p,g,bits=512)
    t0 = time.perf_counter(); sa = dh_shared(p,g,a,B); sb = dh_shared(p,g,b,A); t1=time.perf_counter()
    print("DH shared equal:", sa==sb, "time:", t1-t0, "s")
