# rsa_cia_triage_demo.py
# Educational RSA (toy keygen) + SHA-256 for signature and verification of CIA properties.

import secrets
import hashlib

# ---- Math utils ----
def egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def modinv(a, m):
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m

def is_probable_prime(n, k=16):
    if n < 2: return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n == p: return True
        if n % p == 0: return False
    d = n-1; s = 0
    while d % 2 == 0:
        s += 1; d //= 2
    for _ in range(k):
        a = secrets.randbelow(n-3)+2
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for __ in range(s-1):
            x = (x*x) % n
            if x == n-1: break
        else:
            return False
    return True

def gen_prime(bits=512):
    while True:
        n = secrets.randbits(bits) | 1 | (1 << (bits-1))
        if is_probable_prime(n):
            return n

# ---- RSA keygen, encrypt, decrypt, sign, verify (bare-bones, no padding) ----
def rsa_keygen(bits=1024):
    while True:
        p = gen_prime(bits // 2)
        q = gen_prime(bits // 2)
        if p != q:
            break
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    if phi % e == 0:
        # rare; regenerate
        return rsa_keygen(bits)
    d = modinv(e, phi)
    return (e, n), (d, n)

def i2osp(x: int) -> bytes:
    return x.to_bytes((x.bit_length()+7)//8 or 1, "big")

def os2ip(b: bytes) -> int:
    return int.from_bytes(b, "big")

def rsa_encrypt(pub, m_bytes: bytes) -> int:
    e, n = pub
    m_int = os2ip(m_bytes)
    if m_int >= n:
        raise ValueError("Message too large for RSA modulus (no padding).")
    return pow(m_int, e, n)

def rsa_decrypt(priv, c_int: int) -> bytes:
    d, n = priv
    m_int = pow(c_int, d, n)
    return i2osp(m_int)

def rsa_sign(priv, message: bytes) -> int:
    d, n = priv
    h = hashlib.sha256(message).digest()
    h_int = os2ip(h)
    return pow(h_int, d, n)

def rsa_verify(pub, message: bytes, sig: int) -> bool:
    e, n = pub
    h = hashlib.sha256(message).digest()
    h_int = os2ip(h)
    v = pow(sig, e, n)
    return v == h_int

if __name__ == "__main__":
    # Parties: Sender (S) and Receiver (R)
    pub_R, priv_R = rsa_keygen(1024)  # Receiver's RSA keys (for confidentiality)
    pub_S, priv_S = rsa_keygen(1024)  # Sender's RSA keys (for authenticity)

    message = b"RSA CIA demo: confidentiality (encrypt), integrity+authenticity (sign)."
    print("Plaintext:", message)

    # Confidentiality: encrypt message with Receiver's public key
    c_int = rsa_encrypt(pub_R, message)
    print("Ciphertext (int) length:", c_int.bit_length(), "bits")

    # Integrity + authenticity: Sender signs the message hash
    sig = rsa_sign(priv_S, message)
    print("Signature (int) length:", sig.bit_length(), "bits")

    # Receiver side: decrypt and verify signature
    decrypted = rsa_decrypt(priv_R, c_int)
    ok_sig = rsa_verify(pub_S, decrypted, sig)

    print("Decrypted:", decrypted)
    print("Integrity+Authenticity (signature valid):", ok_sig)

    # Availability note (simple check)
    print("Availability: end-to-end operations completed without error (demo-level).")
