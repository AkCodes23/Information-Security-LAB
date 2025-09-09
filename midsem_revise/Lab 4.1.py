# Lab 4A: Hybrid RSA + DH secure channels among subsystems with simple KMS
# Requires: pycryptodome, cryptography
# Run: python lab4_securecorp.py

import time, secrets
from collections import defaultdict
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.number import getPrime
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

class KMS:
    def __init__(self):
        self.rsa_keys = {}
        self.revoked = set()

    def issue_rsa(self, sys_name, bits=2048):
        priv = RSA.generate(bits)
        self.rsa_keys[sys_name] = priv
        return priv.publickey()

    def revoke(self, sys_name):
        self.revoked.add(sys_name)

    def get_priv(self, sys_name):
        if sys_name in self.revoked: raise PermissionError("Key revoked")
        return self.rsa_keys[sys_name]

def dh_params(bits=512):
    p = getPrime(bits); g = 2
    return p, g

def dh_ephemeral(p,g):
    a = secrets.randbelow(p-2)+2
    A = pow(g,a,p)
    return a,A

def hkdf_key(shared):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"dh").derive(shared.to_bytes((shared.bit_length()+7)//8,'big'))

def channel_establish(p, g, rsa_pub_peer):
    # DH exchange with RSA-authenticated symmetric key wrapping
    a, A = dh_ephemeral(p,g)
    # simulate exchange: send A, receive B (demo uses echo)
    b, B = dh_ephemeral(p,g)
    shared = pow(B, a, p)
    key = hkdf_key(shared)
    # wrap key to peer with RSA
    wrapped = PKCS1_OAEP.new(rsa_pub_peer).encrypt(key)
    return key, wrapped, B, b, A

def unwrap_key(rsa_priv, wrapped):
    return PKCS1_OAEP.new(rsa_priv).decrypt(wrapped)

def encrypt_doc(key, data: bytes):
    iv = get_random_bytes(12)
    c = AES.new(key, AES.MODE_GCM, nonce=iv)
    ct, tag = c.encrypt_and_digest(data)
    return iv, tag, ct

def decrypt_doc(key, package):
    iv, tag, ct = package
    return AES.new(key, AES.MODE_GCM, nonce=iv).decrypt_and_verify(ct, tag)

if __name__ == "__main__":
    kms = KMS()
    # Subsystems
    A_pub = kms.issue_rsa("Finance")
    B_pub = kms.issue_rsa("HR")
    C_pub = kms.issue_rsa("Supply")

    p,g = dh_params()
    # Finance -> HR channel
    kA, wrapped_to_B, B_echo, b, A_echo = channel_establish(p,g,B_pub)
    kB = unwrap_key(kms.get_priv("HR"), wrapped_to_B)
    pkg = encrypt_doc(kA, b"Financial Report Q1")
    print("Finance->HR:", decrypt_doc(kB, pkg).decode())

    # Revocation demo
    kms.revoke("HR")
    try:
        kms.get_priv("HR")
    except Exception as e:
        print("Revoked HR access:", e)
