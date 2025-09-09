import random
import math

# ---------------------------
# 1. Key Generation
# ---------------------------
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def generate_keypair(p, q):
    n = p * q
    lam = lcm(p-1, q-1)
    g = n + 1
    mu = pow(l_function(pow(g, lam, n*n), n), -1, n)
    return (n, g), (lam, mu)

def l_function(x, n):
    return (x - 1) // n

# ---------------------------
# 2. Encryption
# ---------------------------
def encrypt(pub_key, m):
    n, g = pub_key
    n_sq = n * n
    r = random.randint(1, n-1)
    while math.gcd(r, n) != 1:
        r = random.randint(1, n-1)
    c = (pow(g, m, n_sq) * pow(r, n, n_sq)) % n_sq
    return c

# ---------------------------
# 3. Decryption
# ---------------------------
def decrypt(priv_key, pub_key, c):
    n, g = pub_key
    lam, mu = priv_key
    n_sq = n * n
    x = pow(c, lam, n_sq)
    L = l_function(x, n)
    m = (L * mu) % n
    return m

# ---------------------------
# 4. Demonstration
# ---------------------------
if __name__ == "__main__":
    # Small primes for demo (use large primes in real applications)
    p, q = 47, 59
    public_key, private_key = generate_keypair(p, q)

    m1, m2 = 15, 25
    print(f"Original numbers: {m1}, {m2}")

    # Encrypt
    c1 = encrypt(public_key, m1)
    c2 = encrypt(public_key, m2)
    print(f"Ciphertexts: c1 = {c1}, c2 = {c2}")

    # Homomorphic addition: E(m1) * E(m2) mod n^2 = E(m1 + m2)
    n_sq = public_key[0] ** 2
    c_sum = (c1 * c2) % n_sq
    print(f"Encrypted sum (ciphertext): {c_sum}")

    # Decrypt sum
    decrypted_sum = decrypt(private_key, public_key, c_sum)
    print(f"Decrypted sum: {decrypted_sum}")
    print(f"Verification: {decrypted_sum == (m1 + m2)}")
