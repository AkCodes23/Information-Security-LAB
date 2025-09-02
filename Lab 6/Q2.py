# diffie_hellman_demo.py
# Educational DH key exchange, derive symmetric key, encrypt/decrypt with XOR stream.

import secrets
import hashlib

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

def gen_prime(bits=256):
    while True:
        n = secrets.randbits(bits) | 1 | (1 << (bits-1))
        if is_probable_prime(n):
            return n

def find_generator(p):
    while True:
        g = secrets.randbelow(p-3) + 2
        if pow(g, 2, p) != 1 and pow(g, (p-1)//2, p) != 1:
            return g

def derive_key(shared_int: int, length: int) -> bytes:
    # Expand with SHA-256 in counter mode
    out = bytearray()
    counter = 0
    seed = shared_int.to_bytes((shared_int.bit_length()+7)//8 or 1, 'big')
    while len(out) < length:
        out.extend(hashlib.sha256(seed + counter.to_bytes(4,'big')).digest())
        counter += 1
    return bytes(out[:length])

def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

if __name__ == "__main__":
    p = gen_prime(256)
    g = find_generator(p)
    print(f"Group: p({p.bit_length()} bits), g chosen.")

    # Alice
    a = secrets.randbelow(p-2)+1
    A = pow(g, a, p)

    # Bob
    b = secrets.randbelow(p-2)+1
    B = pow(g, b, p)

    # Exchange A, B
    s_alice = pow(B, a, p)
    s_bob   = pow(A, b, p)
    print("Shared equal:", s_alice == s_bob)

    # Symmetric key
    key = derive_key(s_alice, 32)

    msg = b"DH-derived symmetric encryption demo."
    keystream = derive_key(int.from_bytes(key,'big'), len(msg))
    ct = xor_bytes(msg, keystream)
    pt = xor_bytes(ct, keystream)

    print("Ciphertext (hex):", ct.hex())
    print("Decrypted:", pt)
    print("Match:", pt == msg)
