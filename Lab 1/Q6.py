from math import gcd

def modinv(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

def affine_decrypt(text, a, b):
    a_inv = modinv(a, 26)
    if not a_inv:
        return ""
    return ''.join(chr((a_inv * (ord(c) - ord('A') - b)) % 26 + ord('a')) for c in text if c.isalpha())

def matches_known_pair(a, b):
    test = affine_decrypt("GL", a, b)
    return test == "ab"

ciphertext = "XPALASXYFGFUKPXUSOGEUTKCDGEXANMGNVS"
coprimes = [a for a in range(1, 26) if gcd(a, 26) == 1]

for a in coprimes:
    for b in range(26):
        if matches_known_pair(a, b):
            plaintext = affine_decrypt(ciphertext, a, b)
            print(f"Found key: a={a}, b={b}")
            print("Decrypted:", plaintext)
            break
