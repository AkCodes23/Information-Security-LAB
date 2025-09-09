# Lab 1: Attacks helpers (shift identification, affine brute force)
# Run: python lab1_attacks.py

from lab1_classical_ciphers import normalize, caesar_decrypt, affine_decrypt

def identify_shift_cipher(ct, known_plain):
    # Return key if caesar shift of ct equals known_plain
    for k in range(26):
        if normalize(caesar_decrypt(ct, k)) == normalize(known_plain):
            return k
    return None

def brute_force_affine_with_pair(ct, p_pair_pt="ab", p_pair_ct="GL"):
    # Find (a,b) s.t. E('a')='G' and E('b')='L' -> then decrypt ct
    A = ord('a') - ord('a')
    B = ord('b') - ord('a')
    G = ord('G'.lower()) - ord('a')
    L = ord('L'.lower()) - ord('a')
    for a in range(1,26,2):  # a must be coprime to 26; only odds not 13
        if pow(a, -1, 26):
            for b in range(26):
                if (a*A + b) % 26 == G and (a*B + b) % 26 == L:
                    try:
                        pt = affine_decrypt(ct.lower(), a, b)
                        return (a, b, pt)
                    except Exception:
                        pass
    return None

if __name__ == "__main__":
    # Q5: Identify attack and plaintext for "XVIEWYWI" given earlier "CIW"->"yes" shift cipher
    # The type is "crib/known-plaintext" carryover; try all shifts:
    for k in range(26):
        print(k, caesar_decrypt("xviewywi", k))
    # Q6: Affine brute force given 'ab' -> 'GL'
    ct = "XPALASXYFGFUKPXUSOGEUTKCDGEXANMGNVS".lower()
    res = brute_force_affine_with_pair(ct)
    print("Affine key and plaintext:", res)
