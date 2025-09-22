import hashlib
import time
import random
import string
import math
import numpy as np
import matplotlib.pyplot as plt
from Crypto.Cipher import DES, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# COMPREHENSIVE CRYPTOGRAPHY ALGORITHMS IMPLEMENTATION
# Labs 1-6 Complete Implementation with Performance Analysis
# ==============================================================================

print("=" * 80)
print("INFORMATION SECURITY LAB - ALL ALGORITHMS IMPLEMENTATION")
print("Labs 1-6 Complete Coverage with Performance Analysis")
print("=" * 80)

class CryptographyToolkit:
    def __init__(self):
        self.results = {}
        self.timing_data = {}
    
    # ==========================================================================
    # LAB 1: BASIC SYMMETRIC KEY CIPHERS
    # ==========================================================================
    
    def additive_cipher(self, text, key, decrypt=False):
        """Caesar Cipher / Shift Cipher"""
        result = ""
        shift = -key if decrypt else key
        for char in text.upper():
            if char.isalpha():
                result += chr((ord(char) - 65 + shift) % 26 + 65)
            else:
                result += char
        return result
    
    def multiplicative_cipher(self, text, key, decrypt=False):
        """Multiplicative Cipher"""
        result = ""
        if decrypt:
            # Find modular inverse of key
            key_inv = self.mod_inverse(key, 26)
            if key_inv == -1:
                return "Invalid key - no inverse exists"
        else:
            key_inv = key
            
        for char in text.upper():
            if char.isalpha():
                if decrypt:
                    result += chr((key_inv * (ord(char) - 65)) % 26 + 65)
                else:
                    result += chr((key * (ord(char) - 65)) % 26 + 65)
            else:
                result += char
        return result
    
    def affine_cipher(self, text, a, b, decrypt=False):
        """Affine Cipher: E(x) = (ax + b) mod 26"""
        result = ""
        if decrypt:
            a_inv = self.mod_inverse(a, 26)
            if a_inv == -1:
                return "Invalid key 'a' - no inverse exists"
        
        for char in text.upper():
            if char.isalpha():
                x = ord(char) - 65
                if decrypt:
                    result += chr((a_inv * (x - b)) % 26 + 65)
                else:
                    result += chr((a * x + b) % 26 + 65)
            else:
                result += char
        return result
    
    def vigenere_cipher(self, text, key, decrypt=False):
        """Vigenere Cipher"""
        result = ""
        key = key.upper()
        key_index = 0
        
        for char in text.upper():
            if char.isalpha():
                shift = ord(key[key_index % len(key)]) - 65
                if decrypt:
                    shift = -shift
                result += chr((ord(char) - 65 + shift) % 26 + 65)
                key_index += 1
            else:
                result += char
        return result
    
    def playfair_cipher(self, text, key, decrypt=False):
        """Playfair Cipher (Simplified Implementation)"""
        # Create 5x5 matrix
        key = key.upper().replace('J', 'I')
        matrix = []
        used = set()
        
        # Add key letters
        for char in key:
            if char.isalpha() and char not in used:
                matrix.append(char)
                used.add(char)
        
        # Add remaining letters
        for char in 'ABCDEFGHIKLMNOPQRSTUVWXYZ':  # No J
            if char not in used:
                matrix.append(char)
        
        # Convert to 5x5 grid
        grid = [matrix[i:i+5] for i in range(0, 25, 5)]
        
        # Find position of character
        def find_pos(char):
            for i, row in enumerate(grid):
                if char in row:
                    return i, row.index(char)
            return None, None
        
        # Process pairs
        text = text.upper().replace('J', 'I')
        pairs = []
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i] != text[i + 1]:
                pairs.append(text[i:i+2])
                i += 2
            else:
                pairs.append(text[i] + 'X')
                i += 1
        
        result = ""
        for pair in pairs:
            if len(pair) == 2:
                r1, c1 = find_pos(pair[0])
                r2, c2 = find_pos(pair[1])
                
                if r1 is not None and r2 is not None:
                    if r1 == r2:  # Same row
                        if decrypt:
                            result += grid[r1][(c1-1)%5] + grid[r2][(c2-1)%5]
                        else:
                            result += grid[r1][(c1+1)%5] + grid[r2][(c2+1)%5]
                    elif c1 == c2:  # Same column
                        if decrypt:
                            result += grid[(r1-1)%5][c1] + grid[(r2-1)%5][c2]
                        else:
                            result += grid[(r1+1)%5][c1] + grid[(r2+1)%5][c2]
                    else:  # Rectangle
                        result += grid[r1][c2] + grid[r2][c1]
        
        return result
    
    def hill_cipher_2x2(self, text, key_matrix, decrypt=False):
        """Hill Cipher (2x2 matrix)"""
        if decrypt:
            det = (key_matrix[0][0] * key_matrix[1][1] - key_matrix[0][1] * key_matrix[1][0]) % 26
            det_inv = self.mod_inverse(det, 26)
            if det_inv == -1:
                return "Matrix not invertible"
            
            inv_matrix = [
                [(det_inv * key_matrix[1][1]) % 26, (-det_inv * key_matrix[0][1]) % 26],
                [(-det_inv * key_matrix[1][0]) % 26, (det_inv * key_matrix[0][0]) % 26]
            ]
            key_matrix = inv_matrix
        
        result = ""
        text = text.upper().replace(' ', '')
        
        for i in range(0, len(text), 2):
            if i + 1 < len(text):
                pair = [ord(text[i]) - 65, ord(text[i+1]) - 65]
                encrypted = [
                    (key_matrix[0][0] * pair[0] + key_matrix[0][1] * pair[1]) % 26,
                    (key_matrix[1][0] * pair[0] + key_matrix[1][1] * pair[1]) % 26
                ]
                result += chr(encrypted[0] + 65) + chr(encrypted[1] + 65)
        
        return result
    
    # ==========================================================================
    # LAB 2: ADVANCED SYMMETRIC KEY CIPHERS
    # ==========================================================================
    
    def des_encrypt_decrypt(self, data, key, decrypt=False):
        """DES Encryption/Decryption"""
        try:
            cipher = DES.new(key, DES.MODE_ECB)
            if decrypt:
                return cipher.decrypt(data)
            else:
                return cipher.encrypt(pad(data, DES.block_size))
        except Exception as e:
            return f"DES Error: {str(e)}"
    
    def aes_encrypt_decrypt(self, data, key, decrypt=False):
        """AES Encryption/Decryption"""
        try:
            cipher = AES.new(key, AES.MODE_ECB)
            if decrypt:
                return unpad(cipher.decrypt(data), AES.block_size)
            else:
                return cipher.encrypt(pad(data, AES.block_size))
        except Exception as e:
            return f"AES Error: {str(e)}"
    
    # ==========================================================================
    # LAB 3: ASYMMETRIC KEY CIPHERS
    # ==========================================================================
    
    def generate_rsa_keys(self, bits=512):
        """Generate RSA Key Pair (simplified for demo)"""
        # Use small primes for demonstration
        primes = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        p = random.choice(primes)
        q = random.choice([x for x in primes if x != p])
        
        n = p * q
        phi = (p - 1) * (q - 1)
        
        # Choose e
        e = 65537
        while math.gcd(e, phi) != 1:
            e += 2
        
        # Calculate d
        d = self.mod_inverse(e, phi)
        
        return {
            'public': (e, n),
            'private': (d, n),
            'p': p, 'q': q, 'phi': phi
        }
    
    def rsa_encrypt_decrypt(self, message, key, decrypt=False):
        """RSA Encryption/Decryption"""
        if isinstance(message, str):
            message = int.from_bytes(message.encode(), 'big')
        
        exp, n = key
        return pow(message, exp, n)
    
    def elgamal_keygen(self):
        """ElGamal Key Generation (simplified)"""
        p = 2357  # Small prime for demo
        g = 2     # Generator
        x = random.randint(1, p-2)  # Private key
        y = pow(g, x, p)            # Public key
        
        return {
            'public': (p, g, y),
            'private': x
        }
    
    def elgamal_encrypt_decrypt(self, message, keys, decrypt=False):
        """ElGamal Encryption/Decryption"""
        if decrypt:
            # Decrypt: m = (c2 * (c1^x)^-1) mod p
            c1, c2 = message
            x = keys['private']
            p, g, y = keys['public']
            s = pow(c1, x, p)
            s_inv = self.mod_inverse(s, p)
            return (c2 * s_inv) % p
        else:
            # Encrypt
            p, g, y = keys['public']
            k = random.randint(1, p-2)
            c1 = pow(g, k, p)
            c2 = (message * pow(y, k, p)) % p
            return (c1, c2)
    
    # ==========================================================================
    # LAB 5: HASHING
    # ==========================================================================
    
    def simple_hash(self, text):
        """Simple Hash Function"""
        hash_value = 0
        for char in text:
            hash_value = (hash_value * 31 + ord(char)) % 1000000007
        return hash_value
    
    def cryptographic_hashes(self, text):
        """Standard Cryptographic Hash Functions"""
        text_bytes = text.encode('utf-8')
        return {
            'MD5': hashlib.md5(text_bytes).hexdigest(),
            'SHA1': hashlib.sha1(text_bytes).hexdigest(),
            'SHA256': hashlib.sha256(text_bytes).hexdigest(),
            'SHA512': hashlib.sha512(text_bytes).hexdigest()
        }
    
    # ==========================================================================
    # LAB 6: DIGITAL SIGNATURES
    # ==========================================================================
    
    def rsa_sign_verify(self, message, keys, sign=True):
        """RSA Digital Signature"""
        if sign:
            # Sign with private key
            d, n = keys['private']
            hash_val = int(hashlib.sha256(message.encode()).hexdigest(), 16) % n
            signature = pow(hash_val, d, n)
            return signature
        else:
            # Verify with public key (message, signature, keys)
            signature = keys['signature']
            e, n = keys['public']
            hash_val = int(hashlib.sha256(message.encode()).hexdigest(), 16) % n
            decrypted_hash = pow(signature, e, n)
            return hash_val == decrypted_hash
    
    # ==========================================================================
    # UTILITY FUNCTIONS
    # ==========================================================================
    
    def mod_inverse(self, a, m):
        """Extended Euclidean Algorithm for Modular Inverse"""
        if math.gcd(a, m) != 1:
            return -1
        
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, y = extended_gcd(a, m)
        return (x % m + m) % m
    
    def benchmark_algorithm(self, func, *args, iterations=100):
        """Benchmark algorithm performance"""
        times = []
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = func(*args)
                end_time = time.time()
                times.append(end_time - start_time)
            except:
                times.append(0)
        
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times)
        }

# Initialize toolkit
crypto = CryptographyToolkit()

print("\n" + "="*50)
print("TESTING ALL ALGORITHMS")
print("="*50)

# Test data
test_message = "HELLO WORLD"
test_bytes = b"Hello World 1234"
test_key_des = b"12345678"  # 8 bytes for DES
test_key_aes = b"1234567890123456"  # 16 bytes for AES

print(f"\nTest Message: '{test_message}'")
print(f"Test Bytes: {test_bytes}")