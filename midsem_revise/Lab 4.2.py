# Lab 4B: Rabin cryptosystem KMS demo
# Requires: pycryptodome
# Run: python lab4_rabin_kms.py

import time, os
from Crypto.Util.number import getPrime, inverse

def rabin_keypair(bits=512):
    # p,q Blum primes (p%4==3, q%4==3)
    def blum_prime(bits):
        while True:
            p = getPrime(bits)
            if p % 4 == 3:
                return p
    p = blum_prime(bits//2)
    q = blum_prime(bits//2)
    n = p*q
    return (p,q,n)

def rabin_encrypt(n, m: int):
    return pow(m, 2, n)

def rabin_decrypt(p, q, n, c):
    # Four roots; pick canonical via padding convention (demo picks smallest)
    # Compute square roots using CRT
    mp = pow(c, (p+1)//4, p)
    mq = pow(c, (q+1)//4, q)
    yp = inverse(p, q)
    yq = inverse(q, p)
    r1 = (yq*q*mp + yp*p*mq) % n
    r2 = n - r1
    r3 = (yq*q*mp - yp*p*mq) % n
    r4 = n - r3
    return sorted({r1, r2, r3, r4})  # set of four roots

class RabinKMS:
    def __init__(self):
        self.store = {}
        self.revoked = set()

    def generate(self, org, bits=1024):
        p,q,n = rabin_keypair(bits)
        self.store[org] = (p,q,n)
        return n

    def get_priv(self, org):
        if org in self.revoked: raise PermissionError("Revoked")
        return self.store[org][:2]  # p,q

    def get_pub(self, org):
        return self.store[org][2]

    def revoke(self, org):
        self.revoked.add(org)

    def renew(self, org, bits=1024):
        return self.generate(org, bits)

if __name__ == "__main__":
    kms = RabinKMS()
    n = kms.generate("HospitalA", 512)
    p,q = kms.get_priv("HospitalA")
    msg = int.from_bytes(b"Patient Record", "big")
    c = rabin_encrypt(n, msg)
    roots = rabin_decrypt(p,q,n,c)
    # Pick the root that decodes to a sensible prefix (demo: try all)
    for r in roots:
        try:
            print("Try root:", r.to_bytes((r.bit_length()+7)//8, "big"))
        except Exception:
            pass
    kms.revoke("HospitalA")
    try:
        kms.get_priv("HospitalA")
    except Exception as e:
        print("Revoked:", e)

    print("\nTrade-offs (summary):")
    print("- RSA: unique decryption, flexible padding (OAEP), mature tooling.")
    print("- Rabin: decryption yields 4 roots; needs redundancy to disambiguate; faster encryption (square).")
