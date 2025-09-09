import random
import math

# ---------------------------
# Paillier Cryptosystem
# ---------------------------

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def L(u, n):
    return (u - 1) // n

def generate_paillier_keypair(p, q):
    n = p * q
    lam = lcm(p-1, q-1)
    g = n + 1
    mu = pow(L(pow(g, lam, n*n), n), -1, n)
    return (n, g), (lam, mu)

def paillier_encrypt(pub, m):
    n, g = pub
    n2 = n * n
    r = random.randrange(1, n)
    while math.gcd(r, n) != 1:
        r = random.randrange(1, n)
    c = (pow(g, m, n2) * pow(r, n, n2)) % n2
    return c

def paillier_decrypt(priv, pub, c):
    n, g = pub
    lam, mu = priv
    n2 = n * n
    u = pow(c, lam, n2)
    m = (L(u, n) * mu) % n
    return m

def paillier_add_ciphertexts(c1, c2, n):
    """Homomorphic addition: multiply ciphertexts mod n^2."""
    return (c1 * c2) % (n*n)

def paillier_scalar_mul(c, k, n):
    """Multiply encrypted value by scalar k."""
    return pow(c, k, n*n)

# ---------------------------
# Secure Data Sharing Simulation
# ---------------------------
if __name__ == "__main__":
    # Key generation (small primes for demo)
    p, q = 1789, 2027
    pub, priv = generate_paillier_keypair(p, q)
    n, g = pub

    # Parties' private data
    A_val = 46
    B_val = 33

    # Each party encrypts their data
    cA = paillier_encrypt(pub, A_val)
    cB = paillier_encrypt(pub, B_val)

    print(f"Party A's encrypted value: {cA}")
    print(f"Party B's encrypted value: {cB}")

    # Aggregator computes encrypted sum
    c_sum = paillier_add_ciphertexts(cA, cB, n)

    # Aggregator computes weighted sum: 2*A + 3*B
    c_weighted = paillier_add_ciphertexts(
        paillier_scalar_mul(cA, 2, n),
        paillier_scalar_mul(cB, 3, n),
        n
    )

    # Only key-holder decrypts results
    sum_dec = paillier_decrypt(priv, pub, c_sum)
    weighted_dec = paillier_decrypt(priv, pub, c_weighted)

    print(f"\nDecrypted A + B: {sum_dec}   [Expected {A_val + B_val}]")
    print(f"Decrypted 2A + 3B: {weighted_dec}   [Expected {2*A_val + 3*B_val}]")
