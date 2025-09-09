# Lab 1: Classical Ciphers Toolkit + Demos
# Run: python lab1_classical_ciphers.py
# Classical ciphers: additive, multiplicative, affine, Vigenère, autokey, Playfair, Hill
import string
import numpy as np

ALPH = string.ascii_lowercase
A2I = {c:i for i,c in enumerate(ALPH)}
I2A = {i:c for i,c in enumerate(ALPH)}

def normalize(s):
    return "".join(ch.lower() for ch in s if ch.isalpha())

# Additive (Caesar)
def caesar_encrypt(pt, k):
    pt = normalize(pt)
    return "".join(I2A[(A2I[c] + k) % 26] for c in pt)

def caesar_decrypt(ct, k):
    return "".join(I2A[(A2I[c] - k) % 26] for c in ct)

# Multiplicative
def mul_encrypt(pt, k):
    if np.gcd(k, 26) != 1:
        raise ValueError("Key not invertible mod 26")
    pt = normalize(pt)
    return "".join(I2A[(A2I[c] * k) % 26] for c in pt)

def mul_decrypt(ct, k):
    inv = pow(k, -1, 26)
    return "".join(I2A[(A2I[c] * inv) % 26] for c in ct)

# Affine: E(x)= (a x + b) mod 26
def affine_encrypt(pt, a, b):
    if np.gcd(a, 26) != 1:
        raise ValueError("a not invertible mod 26")
    pt = normalize(pt)
    return "".join(I2A[(a*A2I[c] + b) % 26] for c in pt)

def affine_decrypt(ct, a, b):
    inv = pow(a, -1, 26)
    return "".join(I2A[((A2I[c] - b) * inv) % 26] for c in ct)

# Vigenère
def vigenere_encrypt(pt, key):
    pt = normalize(pt)
    key = normalize(key)
    out = []
    for i,c in enumerate(pt):
        k = A2I[key[i % len(key)]]
        out.append(I2A[(A2I[c] + k) % 26])
    return "".join(out)

def vigenere_decrypt(ct, key):
    key = normalize(key)
    out = []
    for i,c in enumerate(ct):
        k = A2I[key[i % len(key)]]
        out.append(I2A[(A2I[c] - k) % 26])
    return "".join(out)

# Autokey (key is integer seed)
def autokey_encrypt(pt, k):
    pt = normalize(pt)
    key_stream = [k] + [A2I[c] for c in pt[:-1]]
    return "".join(I2A[(A2I[c] + key_stream[i]) % 26] for i,c in enumerate(pt))

def autokey_decrypt(ct, k):
    ct = normalize(ct)
    pt_nums = []
    prev = k
    for i, c in enumerate(ct):
        p = (A2I[c] - prev) % 26
        pt_nums.append(p)
        prev = p
    return "".join(I2A[p] for p in pt_nums)

# Playfair
def playfair_key_square(keyword):
    keyword = normalize(keyword).replace('j', 'i')
    seen, sq = set(), []
    for ch in keyword + ALPH:
        if ch == 'j':
            continue
        if ch not in seen:
            seen.add(ch)
            sq.append(ch)
    return [sq[i*5:(i+1)*5] for i in range(5)]

def playfair_find(sq, ch):
    for r in range(5):
        for c in range(5):
            if sq[r][c] == ch:
                return r, c
    raise ValueError

def playfair_prepare(pt):
    pt = normalize(pt).replace('j', 'i')
    dig = []
    i = 0
    while i < len(pt):
        a = pt[i]
        b = pt[i+1] if i+1 < len(pt) else 'x'
        if a == b:
            dig.append((a, 'x'))
            i += 1
        else:
            dig.append((a, b))
            i += 2
    if len(dig[-1]) == 1:
        dig[-1] = (dig[-1][0], 'x')
    return dig

def playfair_encrypt(pt, keyword):
    sq = playfair_key_square(keyword)
    dig = playfair_prepare(pt)
    out = []
    for a,b in dig:
        ra, ca = playfair_find(sq, a)
        rb, cb = playfair_find(sq, b)
        if ra == rb:
            out.append(sq[ra][(ca+1)%5])
            out.append(sq[rb][(cb+1)%5])
        elif ca == cb:
            out.append(sq[(ra+1)%5][ca])
            out.append(sq[(rb+1)%5][cb])
        else:
            out.append(sq[ra][cb])
            out.append(sq[rb][ca])
    return "".join(out)

def playfair_decrypt(ct, keyword):
    sq = playfair_key_square(keyword)
    ct = normalize(ct)
    out = []
    for i in range(0, len(ct), 2):
        a, b = ct[i], ct[i+1]
        ra, ca = playfair_find(sq, a)
        rb, cb = playfair_find(sq, b)
        if ra == rb:
            out.append(sq[ra][(ca-1)%5])
            out.append(sq[rb][(cb-1)%5])
        elif ca == cb:
            out.append(sq[(ra-1)%5][ca])
            out.append(sq[(rb-1)%5][cb])
        else:
            out.append(sq[ra][cb])
            out.append(sq[rb][ca])
    return "".join(out)

# Hill (2x2)
def hill2_encrypt(pt, K):
    pt = normalize(pt)
    if len(pt) % 2 == 1:
        pt += 'x'
    out = []
    for i in range(0, len(pt), 2):
        vec = np.array([[A2I[pt[i]]],[A2I[pt[i+1]]]], dtype=int)
        res = K.dot(vec) % 26
        out.append(I2A[int(res[0,0])])
        out.append(I2A[int(res[1,0])])
    return "".join(out)

def hill2_decrypt(ct, K):
    det = int(round(np.linalg.det(K))) % 26
    det_inv = pow(det, -1, 26)
    K_inv = det_inv * np.round(det * np.linalg.inv(K)).astype(int) % 26
    out = []
    for i in range(0, len(ct), 2):
        vec = np.array([[A2I[ct[i]]],[A2I[ct[i+1]]]], dtype=int)
        res = K_inv.dot(vec) % 26
        out.append(I2A[int(res[0,0])])
        out.append(I2A[int(res[1,0])])
    return "".join(out)

if __name__ == "__main__":
    msg1 = "Iamlearninginformationsecurity"
    print("Additive k=20:", ct1 := caesar_encrypt(msg1, 20), "->", caesar_decrypt(ct1, 20))
    print("Multiplicative k=15:", ct2 := mul_encrypt(msg1, 15), "->", mul_decrypt(ct2, 15))
    print("Affine (15,20):", ct3 := affine_encrypt(msg1, 15, 20), "->", affine_decrypt(ct3, 15, 20))

    msg2 = "thehouseisbeingsoldtonight"
    print("Vigenere 'dollars':", ct4 := vigenere_encrypt(msg2, "dollars"), "->", vigenere_decrypt(ct4, "dollars"))
    print("Autokey k=7:", ct5 := autokey_encrypt(msg2, 7), "->", autokey_decrypt(ct5, 7))

    pf_msg = "The key is hidden under the door pad"
    print("Playfair GUIDANCE:", pf_ct := playfair_encrypt(pf_msg, "GUIDANCE"), "->", playfair_decrypt(pf_ct, "GUIDANCE"))

    K = np.array([[3,3],[2,7]])
    hl_msg = "We live in an insecure world"
    hl_ct = hill2_encrypt(hl_msg, K)
    print("Hill 2x2:", hl_ct, "->", hill2_decrypt(hl_ct, K))
