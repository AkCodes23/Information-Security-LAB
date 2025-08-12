import random

p = 7919
g = 2
h = 6465  # h = g^x mod p
x = 2999  # private key

def message_to_int(msg):
    return int.from_bytes(msg.encode('utf-8'), 'big')

def int_to_message(m):
    return m.to_bytes((m.bit_length() + 7) // 8, 'big').decode('utf-8')

def elgamal_encrypt(m_int, p, g, h):
    k = random.randint(1, p-2)  
    c1 = pow(g, k, p)
    c2 = (m_int * pow(h, k, p)) % p
    return c1, c2

def elgamal_decrypt(c1, c2, p, x):
    s = pow(c1, x, p)
    s_inv = pow(s, -1, p)
    m_int = (c2 * s_inv) % p
    return m_int


message = "Asymmetric Algorithms"
m_int = message_to_int(message)
print(f"Original message as int: {m_int}")


c1, c2 = elgamal_encrypt(m_int, p, g, h)
print(f"Ciphertext: (c1={c1}, c2={c2})")


decrypted_int = elgamal_decrypt(c1, c2, p, x)
decrypted_msg = int_to_message(decrypted_int)

print(f"Decrypted message: {decrypted_msg}")

