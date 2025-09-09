import random

# ---------------------------
# ElGamal over Z_p*
# ---------------------------

def modinv(a, p):
    """Modular inverse using Fermat's little theorem (p must be prime)."""
    return pow(a, p - 2, p)

def elgamal_keygen(p=467, g=2):
    """
    Generate ElGamal keypair.
    p: prime modulus
    g: generator of Z_p*
    """
    x = random.randrange(2, p - 2)  # private key
    y = pow(g, x, p)                # public key component
    return (p, g, y), x

def elgamal_encrypt(pub, m):
    """Encrypt message m with public key pub."""
    p, g, y = pub
    assert 1 <= m < p, "Plaintext must be in [1, p-1]"
    k = random.randrange(2, p - 2)
    a = pow(g, k, p)
    b = (pow(y, k, p) * m) % p
    return (a, b)

def elgamal_decrypt(priv, pub, ct):
    """Decrypt ciphertext ct with private key priv."""
    x = priv
    p, g, y = pub
    a, b = ct
    s = pow(a, x, p)
    return (b * modinv(s, p)) % p

def elgamal_mul_ciphertexts(ct1, ct2, p):
    """
    Homomorphic multiplication of two ciphertexts:
    (a1, b1) * (a2, b2) = (a1*a2 mod p, b1*b2 mod p)
    """
    a1, b1 = ct1
    a2, b2 = ct2
    return ((a1 * a2) % p, (b1 * b2) % p)

# ---------------------------
# Demonstration
# ---------------------------
if __name__ == "__main__":
    # Key generation
    pub, priv = elgamal_keygen()
    p, g, y = pub

    # Messages
    m1, m2 = 7, 3
    print(f"Original: {m1}, {m2}")

    # Encrypt
    c1 = elgamal_encrypt(pub, m1)
    c2 = elgamal_encrypt(pub, m2)
    print(f"Ciphertexts:\n c1 = {c1}\n c2 = {c2}")

    # Homomorphic multiplication
    c_mul = elgamal_mul_ciphertexts(c1, c2, p)
    print(f"Encrypted product (ciphertext): {c_mul}")

    # Decrypt product
    dec = elgamal_decrypt(priv, pub, c_mul)
    print(f"Decrypted product: {dec}")
    print(f"Verification: {dec == (m1 * m2) % p}")
