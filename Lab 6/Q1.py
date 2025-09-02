# elgamal_hybrid_demo.py
# Educational ElGamal (mod prime) + symmetric XOR stream for arbitrary-length messages.

import secrets
import hashlib

# ---------- Math utils ----------
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
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23,29]
    for p in small_primes:
        if n == p: return True
        if n % p == 0: return False
    # write n-1 = 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def gen_prime(bits=256):
    while True:
        n = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if is_probable_prime(n):
            return n

def find_generator(p):
    # Find a generator for Z_p^* by factoring p-1 partially (probabilistic simple try)
    # For demo, try random g until it has large order heuristically.
    # Note: Not guaranteed, but adequate for lab demo.
    while True:
        g = secrets.randbelow(p - 3) + 2
        if pow(g, 2, p) != 1 and pow(g, (p - 1) // 2, p) != 1:
            return g

# ---------- Simple stream cipher (XOR with SHA-256-derived keystream) ----------
def xor_stream(data: bytes, seed_int: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hashlib.sha256(seed_int.to_bytes((seed_int.bit_length()+7)//8 or 1, 'big') + counter.to_bytes(8,'big')).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[:len(data)]))

# ---------- ElGamal hybrid ----------
def elgamal_keygen(bits=256):
    p = gen_prime(bits)
    g = find_generator(p)
    x = secrets.randbelow(p - 2) + 1   # private
    y = pow(g, x, p)                   # public
    return (p, g, y), x

def elgamal_encrypt_session(pub, session_int):
    p, g, y = pub
    r = secrets.randbelow(p - 2) + 1
    c1 = pow(g, r, p)
    s = pow(y, r, p)
    c2 = (session_int * s) % p
    return c1, c2

def elgamal_decrypt_session(priv, pub, c1, c2):
    p, g, y = pub
    x = priv
    s = pow(c1, x, p)
    s_inv = modinv(s, p)
    session_int = (c2 * s_inv) % p
    return session_int

# ---------- Demo ----------
if __name__ == "__main__":
    message = b"ElGamal hybrid demo: confidentiality via ElGamal + XOR stream."
    print(f"Plaintext: {message!r}")

    pub, priv = elgamal_keygen(bits=256)
    p, g, y = pub
    print(f"Public (p bits={p.bit_length()}), g, y generated.")

    # Random session key < p
    session_int = secrets.randbelow(p - 1) + 1

    # Encrypt session key with ElGamal
    c1, c2 = elgamal_encrypt_session(pub, session_int)

    # Symmetric encryption
    ciphertext = xor_stream(message, session_int)

    # Receiver side
    recv_session = elgamal_decrypt_session(priv, pub, c1, c2)
    assert recv_session == session_int, "ElGamal session recovery failed."

    decrypted = xor_stream(ciphertext, recv_session)
    print(f"Decrypted: {decrypted!r}")
    print("Match:", decrypted == message)
