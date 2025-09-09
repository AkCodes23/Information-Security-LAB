import math
import random

# ---------------------------
# 1. Key Generation
# ---------------------------
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def modinv(a, m):
    # Extended Euclidean Algorithm
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        a, m = m, a % m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def generate_keypair(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)

    # Choose e
    e = random.randrange(2, phi)
    while gcd(e, phi) != 1:
        e = random.randrange(2, phi)

    # Compute d
    d = modinv(e, phi)
    return (e, n), (d, n)

# ---------------------------
# 2. Encryption
# ---------------------------
def encrypt(pub_key, plaintext):
    e, n = pub_key
    return pow(plaintext, e, n)

# ---------------------------
# 3. Decryption
# ---------------------------
def decrypt(priv_key, ciphertext):
    d, n = priv_key
    return pow(ciphertext, d, n)

# ---------------------------
# 4. Demonstration
# ---------------------------
if __name__ == "__main__":
    # Small primes for demo (use large primes in real applications)
    p, q = 61, 53
    public_key, private_key = generate_keypair(p, q)

    m1, m2 = 7, 3
    print(f"Original numbers: {m1}, {m2}")

    # Encrypt
    c1 = encrypt(public_key, m1)
    c2 = encrypt(public_key, m2)
    print(f"Ciphertexts: c1 = {c1}, c2 = {c2}")

    # Homomorphic multiplication: E(m1) * E(m2) mod n = E(m1 * m2)
    n = public_key[1]
    c_product = (c1 * c2) % n
    print(f"Encrypted product (ciphertext): {c_product}")

    # Decrypt product
    decrypted_product = decrypt(private_key, c_product)
    print(f"Decrypted product: {decrypted_product}")
    print(f"Verification: {decrypted_product == (m1 * m2)}")
