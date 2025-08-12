import random

def mod_exp(base, exp, mod):
    return pow(base, exp, mod)

def elgamal_keygen(p, g):
    x = random.randint(1, p-2)  # Private key
    h = mod_exp(g, x, p)        # Public key component
    return (p, g, h), x

def elgamal_encrypt(pub_key, message_int):
    p, g, h = pub_key
    y = random.randint(1, p-2)
    c1 = mod_exp(g, y, p)
    s = mod_exp(h, y, p)
    c2 = (message_int * s) % p
    return (c1, c2)

def elgamal_decrypt(priv_key, cipher, p):
    c1, c2 = cipher
    s = mod_exp(c1, priv_key, p)
    s_inv = pow(s, -1, p)
    message_int = (c2 * s_inv) % p
    return message_int

# Parameters (p must be large prime)
p = 30803  # Example small prime for demonstration
g = 2

# Key generation
public_key, private_key = elgamal_keygen(p, g)

message = "Confidential Data"
message_int = int.from_bytes(message.encode(), 'big')

# Encrypt
ciphertext = elgamal_encrypt(public_key, message_int)
print("Ciphertext:", ciphertext)

# Decrypt
decrypted_int = elgamal_decrypt(private_key, ciphertext, p)
decrypted_msg = decrypted_int.to_bytes((decrypted_int.bit_length() + 7) // 8, 'big').decode()
print("Decrypted message:", decrypted_msg)
