import random
import math

# ---------------------------
# Paillier cryptosystem (additively homomorphic)
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
    return (pow(g, m, n2) * pow(r, n, n2)) % n2

def paillier_decrypt(priv, pub, c):
    n, g = pub
    lam, mu = priv
    n2 = n * n
    u = pow(c, lam, n2)
    return (L(u, n) * mu) % n

def paillier_add_ciphertexts(c1, c2, n):
    return (c1 * c2) % (n*n)

# ---------------------------
# Shamir secret sharing (t-of-n) over a prime field
# ---------------------------

def modinv(a, p):
    return pow(a, p-2, p)

def shamir_split(secret, t, n, P):
    """
    Split 'secret' into n shares with threshold t over field P (prime).
    Returns list of (i, share_i) for i=1..n
    """
    coeffs = [secret] + [random.randrange(0, P) for _ in range(t-1)]
    shares = []
    for i in range(1, n+1):
        x = i
        y = 0
        xp = 1
        for a in coeffs:
            y = (y + a * xp) % P
            xp = (xp * x) % P
        shares.append((i, y))
    return shares

def shamir_reconstruct(subset_shares, P):
    """
    Reconstruct secret from subset of shares using Lagrange interpolation at x=0.
    subset_shares: list of (i, share_i)
    """
    secret = 0
    for j, yj in subset_shares:
        # Lagrange basis L_j(0)
        num, den = 1, 1
        for m, ym in subset_shares:
            if m == j:
                continue
            num = (num * (-m % P)) % P
            den = (den * (j - m)) % P
        Lj0 = (num * modinv(den % P, P)) % P
        secret = (secret + yj * Lj0) % P
    return secret

# ---------------------------
# Secure thresholding demo
# ---------------------------

def secure_threshold_count_demo():
    # 1) Setup Paillier (small demo primes; use large primes in practice)
    p_paillier, q_paillier = 1789, 2027
    pub, priv = generate_paillier_keypair(p_paillier, q_paillier)
    n, g = pub
    lam, mu = priv

    # 2) Split the Paillier private key into t-of-n shares (educational, not production threshold Paillier)
    # We share lambda and mu separately over a large field P (P must exceed both lambda and mu)
    P_field = 2**127 - 1  # a large Mersenne-like prime for the demo field
    t, n_parties = 3, 5   # require any 3 of 5 parties to collaborate

    lam_shares = shamir_split(lam % P_field, t, n_parties, P_field)
    mu_shares  = shamir_split(mu  % P_field, t, n_parties, P_field)

    # 3) Parties' private data and local thresholding
    values = [42, 57, 61, 39, 70, 12, 58]  # example data across many users
    T = 50
    # Each party locally computes a bit (>= T) and encrypts it
    encrypted_bits = [paillier_encrypt(pub, 1 if v >= T else 0) for v in values]

    # 4) Aggregator computes encrypted count using homomorphic addition
    n2 = n * n
    C = 1
    for c in encrypted_bits:
        C = (C * c) % n2

    # 5) Threshold-controlled decryption:
    # Any t parties pool their Shamir shares to reconstruct lambda and mu, then decrypt.
    # (In true threshold Paillier, no single place reconstructs the whole key; this is an educational stand-in.)
    chosen = random.sample(range(n_parties), t)
    lam_subset = [lam_shares[i] for i in chosen]
    mu_subset  = [mu_shares[i]  for i in chosen]

    lam_rec = shamir_reconstruct(lam_subset, P_field)
    mu_rec  = shamir_reconstruct(mu_subset,  P_field)

    # Ensure they match originals modulo P_field
    assert lam_rec % P_field == lam % P_field
    assert mu_rec  % P_field == mu  % P_field

    # Use reconstructed key to decrypt the aggregated result
    count = paillier_decrypt((lam_rec, mu_rec), pub, C)

    print("Secure thresholding with Paillier (educational threshold):")
    print(f"- Parties: {len(values)} values, threshold T = {T}")
    print(f"- Encrypted count decrypted by any {t} of {n_parties} key-share holders")
    print(f"- Count of values >= {T}: {count}  [Expected {sum(1 for v in values if v >= T)}]")

if __name__ == "__main__":
    secure_threshold_count_demo()
