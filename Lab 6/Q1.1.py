# schnorr_signature_demo.py
# Educational Schnorr signature over a subgroup of Z_p^* (not production secure).

import secrets
import hashlib

# ----- Math utils -----
def egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def is_probable_prime(n, k=16):
    if n < 2:
        return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n == p: return True
        if n % p == 0: return False
    d = n - 1; s = 0
    while d % 2 == 0:
        s += 1; d //= 2
    import secrets
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

def gen_prime(bits):
    import secrets
    while True:
        n = secrets.randbits(bits) | 1 | (1 << (bits-1))
        if is_probable_prime(n):
            return n

def sha256_int(*parts: bytes) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest(), 'big')

# ----- Parameter generation: find q | (p-1) and generator g of order q -----
def schnorr_params(q_bits=160, slack=256):
    # Find q prime, then search for k s.t. p = k*q + 1 is prime, then derive g of order q
    q = gen_prime(q_bits)
    # Search small k for demo speed
    k = 2
    while True:
        p = k*q + 1
        if is_probable_prime(p):
            break
        k += 1
    # Find h s.t. g = h^((p-1)/q) mod p != 1
    while True:
        h = secrets.randbelow(p-3) + 2
        g = pow(h, (p-1)//q, p)
        if g != 1:
            return p, q, g

# ----- Keys, sign, verify -----
def keygen(p, q, g):
    x = secrets.randbelow(q-1) + 1  # private
    y = pow(g, x, p)                # public
    return x, y

def sign(p, q, g, x, message: bytes):
    r = secrets.randbelow(q-1) + 1
    a = pow(g, r, p)
    e = sha256_int(message, a.to_bytes((p.bit_length()+7)//8, 'big')) % q
    s = (r + x*e) % q
    return (e, s)

def verify(p, q, g, y, message: bytes, sig):
    e, s = sig
    # a' = g^s * y^{-e} mod p
    y_inv_e = pow(pow(y, e, p), p-2, p)
    a_prime = (pow(g, s, p) * y_inv_e) % p
    e_prime = sha256_int(message, a_prime.to_bytes((p.bit_length()+7)//8, 'big')) % q
    return e_prime == e

if __name__ == "__main__":
    p, q, g = schnorr_params()
    print(f"Params: p({p.bit_length()} bits), q({q.bit_length()} bits)")

    x, y = keygen(p, q, g)
    message = b"Schnorr signature lab demo."
    sig = sign(p, q, g, x, message)
    ok = verify(p, q, g, y, message, sig)

    print("Signature valid:", ok)
    # Negative test
    ok_tamper = verify(p, q, g, y, message + b"!", sig)
    print("Valid after tamper (should be False):", ok_tamper)
