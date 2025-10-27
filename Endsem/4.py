# PYTHON SCRIPT (Version 6 - Added More Combinations & PKSE/SSE Placeholders)
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
    ICT3141 ULTIMATE EXAM SCRIPT - ALL SCENARIOS V6
    
    Part 1: 12 Base 3-Algorithm Systems (Systems 1-12)
    Part 2: Interactive Classical Ciphers (Systems 13-20)
    Part 3: Exam-Specific & New Combinations (Systems 21-28)
    Part 4: Advanced Concepts (Placeholders) (Systems 29-30)
    Part 5: Tools & Exit (Systems 31+)
    
    All with Menu-Driven Interfaces and Performance Graphs
═══════════════════════════════════════════════════════════════════════════
"""

# --- Imports and Library Checks (Same as V5 script) ---
import sys
import random
import time
import hashlib
import base64
import json
from datetime import datetime
from collections import defaultdict
import math
import os

print("\n" + "="*70)
print("  ICT3141 ULTIMATE EXAM SCRIPT V6 - INITIALIZING")
print("="*70)

# --- PyCryptodome ---
try:
    from Crypto.Cipher import DES, DES3, AES, PKCS1_OAEP
    from Crypto.PublicKey import RSA, ElGamal
    from Crypto.Hash import SHA256, SHA512, SHA1, MD5
    from Crypto.Util import number
    from Crypto.Util.Padding import pad, unpad
    from Crypto.Random import get_random_bytes
    from Crypto.Signature import pkcs1_15
    HAS_CRYPTO = True
    print("  ✓ PyCryptodome loaded")
except ImportError:
    HAS_CRYPTO = False
    print("  ✗ PyCryptodome not installed!")

# --- NumPy ---
try:
    import numpy as np
    HAS_NUMPY = True
    print("  ✓ NumPy loaded")
except ImportError:
    HAS_NUMPY = False
    print("  ⚠ NumPy not available (Hill Cipher disabled)")

# --- PHE ---
try:
    from phe import paillier
    HAS_PAILLIER = True
    print("  ✓ Paillier (phe) loaded")
except ImportError:
    HAS_PAILLIER = False
    print("  ⚠ Paillier (phe) not available (Homomorphic systems disabled)")

# --- Matplotlib ---
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("  ✓ Matplotlib loaded")
except ImportError:
    HAS_MATPLOTLIB = False
    print("  ⚠ Matplotlib not available (Using ASCII graphs)")

print("="*70 + "\n")


# --- Utility Functions (PerformanceTracker, gcd, mod_inverse, matrix_mod_inv, ElGamal Signatures, clear_screen, pause - Same as V5 script) ---
# PASTE THE FULL UTILITY FUNCTIONS CODE FROM VERSION 5 HERE
class PerformanceTracker:
    """Track and visualize performance metrics with enhanced graphing"""
    # (Code from V5 - including matplotlib option)
    def __init__(self): self.metrics = []
    def record(self, operation, time_taken, data_size=0): self.metrics.append({'operation': operation, 'time': time_taken,'size': data_size, 'timestamp': datetime.now().isoformat()})
    def get_stats(self, operation=None):
        data = [m for m in self.metrics if not operation or m['operation'] == operation];
        if not data: return None
        times = [m['time'] for m in data]; sizes = [m['size'] for m in data if m['size'] > 0]; avg_size = sum(sizes) / len(sizes) if sizes else 0
        return {'count': len(times), 'average': sum(times) / len(times),'min': min(times), 'max': max(times), 'total': sum(times),'avg_size': avg_size}
    def print_graph(self, title="Performance Analysis"):
        print(f"\n{'='*70}\n  📊 {title}\n{'='*70}"); ops = defaultdict(list); [ops[m['operation']].append(m['time']) for m in self.metrics]
        if not ops: print("  No data recorded yet"); print("="*70); return
        print("\nOperation Statistics:"); stats_list = []
        for op, times in sorted(ops.items()):
            avg = sum(times) / len(times); stats_list.append({'op': op, 'avg': avg, 'count': len(times), 'min': min(times), 'max': max(times)})
            print(f"\n  {op}:\n    Count:   {len(times)}\n    Average: {avg:.6f}s\n    Min:     {min(times):.6f}s\n    Max:     {max(times):.6f}s")
            scale = 2000; bar = '█' * int(avg * scale); bar = bar if bar else ('·' if avg > 0 else ''); print(f"    Graph:   {bar}")
        print("\n" + "="*70)
        if HAS_MATPLOTLIB:
            try:
                labels = [s['op'] for s in stats_list]; averages = [s['avg'] for s in stats_list]
                plt.figure(figsize=(10, max(6, len(labels)*0.5))); plt.barh(labels, averages, color='skyblue')
                plt.xlabel("Average Time (seconds)"); plt.ylabel("Operation"); plt.title(title); plt.tight_layout(); print("\n📈 Displaying graphical performance chart..."); plt.show()
            except Exception as e: print(f"  ⚠ Matplotlib graph error: {e}")
    def compare_algorithms(self, alg_list):
        if len(alg_list) < 2: print("Need >= 2 algorithms."); return
        stats = {alg: s['average'] for alg in alg_list if (s := self.get_stats(alg))};
        if len(stats) < 2: print("Not enough data."); return
        print(f"\n{'='*70}\n  🚀 ALGORITHM COMPARISON\n{'='*70}"); sorted_algs = sorted(stats.items(), key=lambda x: x[1])
        print("\nSpeed Ranking (fastest to slowest):"); scale = 5000; max_len = max(len(alg) for alg, t in sorted_algs)
        for i, (alg, time_val) in enumerate(sorted_algs, 1):
            bar = '█' * int(time_val * scale); bar = bar if bar else ('·' if time_val > 0 else ''); print(f"  {i}. {alg:<{max_len}s}: {time_val:.8f}s\n     {bar}")
        fastest_time = sorted_algs[0][1]
        if fastest_time > 0: print("\nRelative Performance:"); [print(f"  {alg} is {time_val / fastest_time:.1f}x slower than {sorted_algs[0][0]}") for alg, time_val in sorted_algs[1:]]
        else: print("\nFastest too fast for relative comparison.")
        print("="*70)
        if HAS_MATPLOTLIB:
             try:
                 labels = [a[0] for a in sorted_algs]; times = [a[1] for a in sorted_algs]
                 plt.figure(figsize=(10, max(6, len(labels)*0.5))); plt.barh(labels, times, color='lightcoral'); plt.xlabel("Average Time (seconds)"); plt.ylabel("Algorithm"); plt.title("Algorithm Speed Comparison"); plt.tight_layout(); print("\n📈 Displaying graphical comparison chart..."); plt.show()
             except Exception as e: print(f"  ⚠ Matplotlib comparison graph error: {e}")

def gcd(a, b): return math.gcd(a, b)
def mod_inverse(a, m):
    a=a%m; m0=m; x0,x1=0,1;
    if gcd(a,m)!=1: return None
    while a>1: q=a//m; m,a=a%m,m; x0,x1=x1-q*x0,x0
    if x1<0: x1+=m0
    return x1
def matrix_mod_inv(matrix, modulus):
    if not HAS_NUMPY: raise ImportError("NumPy required")
    det = int(np.round(np.linalg.det(matrix))); det_inv = mod_inverse(det % modulus, modulus)
    if det_inv is None: raise ValueError("Matrix not invertible")
    adj = np.linalg.inv(matrix) * det; inv = (det_inv * np.round(adj)) % modulus
    return inv.astype(int)
def generate_elgamal_sig_keys(bits=1024):
    if not HAS_CRYPTO: raise ImportError("PyCryptodome required")
    while True: q = number.getPrime(bits - 1); p = 2 * q + 1;
        if number.isPrime(p): break
    while True: g_cand = number.getRandomRange(2, p - 1);
        if pow(g_cand, q, p) == 1 and pow(g_cand, 2, p) != 1: g = g_cand; break
    x = number.getRandomRange(2, q); y = pow(g, x, p)
    return {'p': p, 'q': q, 'g': g, 'x': x, 'y': y}
def elgamal_sign(msg_bytes, priv_key):
    if not HAS_CRYPTO: raise ImportError("PyCryptodome required")
    p, q, g, x = priv_key['p'], priv_key['q'], priv_key['g'], priv_key['x']; h_obj = SHA256.new(msg_bytes); h_int = int.from_bytes(h_obj.digest(), 'big')
    while True:
        k = number.getRandomRange(2, q); r = pow(g, k, p); k_inv = number.inverse(k, q); s = (k_inv * (h_int + x * r)) % q
        if r != 0 and s != 0: return int(r), int(s)
def elgamal_verify(msg_bytes, signature, pub_key):
    if not HAS_CRYPTO: raise ImportError("PyCryptodome required")
    p, q, g, y = pub_key['p'], pub_key['q'], pub_key['g'], pub_key['y']; r, s = signature
    if not (0 < r < p and 0 < s < q): return False
    h_obj = SHA256.new(msg_bytes); h_int = int.from_bytes(h_obj.digest(), 'big')
    v1 = (pow(y, r, p) * pow(r, s, p)) % p; v2 = pow(g, h_int, p)
    return v1 == v2
def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')
def pause(): input("\nPress Enter to continue...")
# Ensure CryptographyToolkit class is defined for classical ciphers
class CryptographyToolkit: # Define the class used by classical menus
     def additive_cipher(self, text, key, decrypt=False): shift = -key if decrypt else key; return "".join([chr((ord(c)-65+shift)%26+65) if c.isalpha() else c for c in text.upper()])
     def multiplicative_cipher(self, text, key, decrypt=False):
         key_op = mod_inverse(key, 26) if decrypt else key
         if key_op is None or key_op == -1: return "Invalid key/inverse"
         return "".join([chr((key_op*(ord(c)-65))%26+65) if c.isalpha() else c for c in text.upper()])
     def affine_cipher(self, text, a, b, decrypt=False):
         a_op = mod_inverse(a, 26) if decrypt else a
         if a_op is None or a_op == -1: return "Invalid 'a'/inverse"
         return "".join([chr((a_op*((ord(c)-65) - (b if decrypt else 0)))%26 + 65) if c.isalpha() else c for c in text.upper()]) if decrypt else "".join([chr((a*(ord(c)-65)+b)%26+65) if c.isalpha() else c for c in text.upper()])
     def vigenere_cipher(self, text, key, decrypt=False):
         res=""; k=key.upper(); ki=0
         for c in text.upper():
             if c.isalpha(): shift = ord(k[ki % len(k)]) - 65; shift = -shift if decrypt else shift; res += chr((ord(c)-65+shift)%26+65); ki+=1
             else: res += c
         return res
     def autokey_cipher(self, text, key, decrypt=False): # Simplified Autokey from V5
        res = ""; k = key.upper(); extended_key = k
        if not decrypt: extended_key += text.upper().replace(' ', '')
        ki = 0
        for char in text.upper():
            if char.isalpha():
                if ki < len(extended_key): shift = ord(extended_key[ki]) - 65
                else: shift = 0 # Should not happen if extended correctly
                if decrypt:
                    dec_char_ord = (ord(char) - 65 - shift) % 26
                    dec_char = chr(dec_char_ord + 65)
                    res += dec_char
                    if ki >= len(k): extended_key += dec_char # Extend key during decryption
                else: res += chr((ord(char) - 65 + shift) % 26 + 65)
                ki += 1
            else: res += char
        return res
     def playfair_cipher(self, text, key, decrypt=False): # Simplified Playfair from V5
        key=key.upper().replace('J','I'); matrix=[]; used=set(); [ (matrix.append(c), used.add(c)) for c in key if c.isalpha() and c not in used ]; [ matrix.append(c) for c in 'ABCDEFGHIKLMNOPQRSTUVWXYZ' if c not in used ]; grid=[matrix[i:i+5] for i in range(0,25,5)];
        def find_pos(char):
             for r, row in enumerate(grid):
                  if char in row: return r, row.index(char)
             return None, None
        text=text.upper().replace('J','I').replace(' ',''); pairs=[]; i=0
        while i<len(text): pairs.append(text[i]+'X') if i+1>=len(text) or text[i]==text[i+1] else pairs.append(text[i:i+2]); i+=(1 if i+1>=len(text) or text[i]==text[i+1] else 2)
        res=""; shift = -1 if decrypt else 1
        for p in pairs:
            if len(p)==2: r1,c1=find_pos(p[0]); r2,c2=find_pos(p[1])
            if r1 is not None and r2 is not None:
                 if r1==r2: res+=grid[r1][(c1+shift)%5]+grid[r2][(c2+shift)%5]
                 elif c1==c2: res+=grid[(r1+shift)%5][c1]+grid[(r2+shift)%5][c2]
                 else: res+=grid[r1][c2]+grid[r2][c1]
        return res
     def hill_cipher_2x2(self, text, key_matrix, decrypt=False): # Simplified Hill from V5
        if decrypt: det = (key_matrix[0][0]*key_matrix[1][1]-key_matrix[0][1]*key_matrix[1][0])%26; det_inv = mod_inverse(det, 26);
            if det_inv is None: return "Matrix not invertible"
            inv_matrix = [[(det_inv*key_matrix[1][1])%26, (-det_inv*key_matrix[0][1])%26],[(-det_inv*key_matrix[1][0])%26, (det_inv*key_matrix[0][0])%26]]; key_matrix = inv_matrix
        res = ""; txt = text.upper().replace(' ',''); txt += 'X' * (len(txt)%2)
        for i in range(0,len(txt),2):
             p=[ord(txt[i])-65, ord(txt[i+1])-65]; e=[(key_matrix[0][0]*p[0]+key_matrix[0][1]*p[1])%26, (key_matrix[1][0]*p[0]+key_matrix[1][1]*p[1])%26]; res+=chr(e[0]+65)+chr(e[1]+65)
        return res
toolkit_classical = CryptographyToolkit() # Instantiate for use in menus


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 1: COMPLETE 3-ALGORITHM SYSTEMS (UNCHANGED from Script 1)
# Systems 1-12 - PLACEHOLDERS - PASTE YOUR FULL CODE HERE
#
# ═══════════════════════════════════════════════════════════════════════════
# --- PASTE FULL CODE FOR SYSTEMS 1-12 HERE ---
# (Using placeholder functions as before)
def menu_email_system(): print("\n[Placeholder Sys 1: DES+RSA+SHA256]\nPASTE CODE HERE"); pause()
def menu_banking_system(): print("\n[Placeholder Sys 2: AES+ElGamal(Enc)+SHA512]\nPASTE CODE HERE"); pause()
def menu_cloud_system(): print("\n[Placeholder Sys 3: Rabin+RSA+MD5]\nPASTE CODE HERE"); pause()
def menu_legacy_banking(): print("\n[Placeholder Sys 4: 3DES+ElGamal(Enc)+SHA1]\nPASTE CODE HERE"); pause()
def menu_healthcare(): print("\n[Placeholder Sys 5: AES+RSA+SHA256]\nPASTE CODE HERE"); pause()
def menu_document_management(): print("\n[Placeholder Sys 6: DES+ElGamal(Enc)+MD5]\nPASTE CODE HERE"); pause()
def menu_messaging(): print("\n[Placeholder Sys 7: AES+ElGamal(Enc)+MD5]\nPASTE CODE HERE"); pause()
def menu_file_transfer(): print("\n[Placeholder Sys 8: DES+RSA+SHA512]\nPASTE CODE HERE"); pause()
def menu_digital_library(): print("\n[Placeholder Sys 9: Rabin+ElGamal(Enc)+SHA256]\nPASTE CODE HERE"); pause()
def menu_secure_chat(): print("\n[Placeholder Sys 10: AES+RSA+SHA512]\nPASTE CODE HERE"); pause()
def menu_voting_system():
    if HAS_PAILLIER: print("\n[Placeholder Sys 11: Paillier+ElGamal(Sig)+SHA256]\nPASTE CODE HERE"); pause()
    else: print("\nSys 11 requires 'phe'."); pause()
def menu_hill_hybrid():
    if HAS_NUMPY: print("\n[Placeholder Sys 12: Hill+RSA+SHA256]\nPASTE CODE HERE"); pause()
    else: print("\nSys 12 requires 'numpy'."); pause()

# ═══════════════════════════════════════════════════════════════════════════
#
# PART 2: INTERACTIVE CLASSICAL CIPHERS
# Systems 13-20 (Menus call implementations in toolkit_classical)
#
# ═══════════════════════════════════════════════════════════════════════════
# --- PASTE INTERACTIVE MENUS 13-20 FROM V5 SCRIPT HERE ---
# (Using menus from V5 script)
def menu_additive_cipher(): # System 13
    clear_screen(); print("🔤 SYSTEM 13: ADDITIVE CIPHER (CAESAR)"); print("="*40); msg = input("Msg: ").upper();
    try: key = int(input("Key (0-25): ")) % 26; enc = toolkit_classical.additive_cipher(msg, key); dec = toolkit_classical.additive_cipher(enc, key, True); print(f"\nOrig: {msg}\nEnc: {enc}\nDec: {dec}"); pause()
    except ValueError: print("Invalid key."); pause()
def menu_multiplicative_cipher(): # System 14
    clear_screen(); print("🔢 SYSTEM 14: MULTIPLICATIVE CIPHER"); print("="*40); msg = input("Msg: ").upper(); print("Valid keys: 1,3,5,7,9,11,15,17,19,21,23,25");
    try: key = int(input("Key: "));
        if gcd(key, 26) != 1: raise ValueError("Key not coprime to 26")
        enc = toolkit_classical.multiplicative_cipher(msg, key); dec = toolkit_classical.multiplicative_cipher(enc, key, True); print(f"\nOrig: {msg}\nEnc: {enc}\nDec: {dec}\nInv: {mod_inverse(key, 26)}"); pause()
    except ValueError as e: print(f"Invalid key: {e}"); pause()
def menu_affine_cipher(): # System 15
    clear_screen(); print("🔡 SYSTEM 15: AFFINE CIPHER"); print("="*30); msg = input("Msg: ").upper();
    try: a = int(input("a (coprime to 26): ")); b = int(input("b (0-25): ")) % 26;
        if gcd(a, 26) != 1: raise ValueError("'a' not coprime to 26")
        enc = toolkit_classical.affine_cipher(msg, a, b); dec = toolkit_classical.affine_cipher(enc, a, b, True); print(f"\nOrig: {msg}\nEnc: {enc}\nDec: {dec}\nParams: a={a}, b={b}, a_inv={mod_inverse(a, 26)}"); pause()
    except ValueError as e: print(f"Invalid key(s): {e}"); pause()
def menu_vigenere_cipher(): # System 16
    clear_screen(); print("🔠 SYSTEM 16: VIGENERE CIPHER"); print("="*30); msg = input("Msg: ").upper(); key = input("Key: ").upper();
    if not key.isalpha(): print("Invalid key."); pause(); return
    enc = toolkit_classical.vigenere_cipher(msg, key); dec = toolkit_classical.vigenere_cipher(enc, key, True); print(f"\nOrig: {msg}\nKey:  {key}\nEnc: {enc}\nDec: {dec}"); pause()
def menu_autokey_cipher(): # System 17
    clear_screen(); print("🔑 SYSTEM 17: AUTOKEY CIPHER"); print("="*30); msg = input("Msg: ").upper(); key = input("Key: ").upper();
    if not key.isalpha(): print("Invalid key."); pause(); return
    enc = toolkit_classical.autokey_cipher(msg, key); dec = toolkit_classical.autokey_cipher(enc, key, True); print(f"\nOrig: {msg}\nKey:  {key}\nEnc: {enc}\nDec: {dec}"); pause()
def menu_playfair_cipher(): # System 18
    clear_screen(); print("▦ SYSTEM 18: PLAYFAIR CIPHER"); print("="*30); msg = input("Msg: ").upper(); key = input("Key: ").upper();
    if not key.isalpha(): print("Invalid key."); pause(); return
    enc = toolkit_classical.playfair_cipher(msg, key); dec = toolkit_classical.playfair_cipher(enc, key, True); print(f"\nOrig: {msg}\nKey:  {key}\nEnc: {enc}\nDec: {dec}"); pause()
def menu_hill_cipher(): # System 19
    if not HAS_NUMPY: print("\nHill Cipher requires NumPy."); pause(); return
    clear_screen(); print("🔢 SYSTEM 19: HILL CIPHER (2x2)"); print("="*30); msg = input("Msg: ").upper(); key_str = input("Key (e.g., '3 3 2 7'): ").split()
    try: key_vals = [int(v) for v in key_str];
        if len(key_vals)!=4: raise ValueError("Need 4 values"); key_matrix = [key_vals[0:2], key_vals[2:4]]; det = (key_matrix[0][0]*key_matrix[1][1]-key_matrix[0][1]*key_matrix[1][0])%26;
        if mod_inverse(det, 26) is None: raise ValueError(f"Matrix det={det} not invertible")
        enc = toolkit_classical.hill_cipher_2x2(msg, key_matrix); dec = toolkit_classical.hill_cipher_2x2(enc, key_matrix, True); print(f"\nOrig: {msg}\nKey: {key_matrix}\nEnc: {enc}\nDec: {dec}"); pause()
    except ValueError as e: print(f"Invalid key: {e}"); pause()
def menu_transposition_cipher(): # System 20
    # (Code from V5 - transposition needs careful review for perfect decryption)
    clear_screen(); print("🔄 SYSTEM 20: TRANSPOSITION CIPHER (COLUMNAR)"); print("="*50); message = input("Msg: ").upper().replace(" ", ""); key = input("Key: ").upper();
    if not key.isalpha() or len(key) == 0: print("Invalid key."); pause(); return
    def transpose_encrypt(msg, keyword): # Simple columnar transpose
        key_order=sorted([(c,i) for i,c in enumerate(keyword)]); num_cols=len(keyword); num_rows=math.ceil(len(msg)/num_cols); grid=[['X'] * num_cols for _ in range(num_rows)]; msg_idx=0
        for r in range(num_rows):
             for c in range(num_cols):
                 if msg_idx < len(msg): grid[r][c]=msg[msg_idx]; msg_idx+=1
        return "".join( grid[r][c] for _,c in key_order for r in range(num_rows) )
    def transpose_decrypt(cipher, keyword): # Basic decrypt assumes full grid
        key_order=sorted([(c,i) for i,c in enumerate(keyword)]); num_cols=len(keyword); num_rows=math.ceil(len(cipher)/num_cols); grid=[[''] * num_cols for _ in range(num_rows)]; cipher_idx=0
        for _,c in key_order:
            for r in range(num_rows):
                if cipher_idx < len(cipher): grid[r][c]=cipher[cipher_idx]; cipher_idx+=1
        return "".join( grid[r][c] for r in range(num_rows) for c in range(num_cols) ).rstrip('X') # Basic padding removal
    enc = transpose_encrypt(message, key); dec = transpose_decrypt(encrypted, key); print(f"\nOrig: {message}\nKey:  {key}\nEnc: {enc}\nDec: {dec} (Padding removal basic)"); pause()


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 3: EXAM-SPECIFIC SCENARIOS & NEW COMBINATIONS
# Systems 21-28 (Includes 21-26 from V5, plus 27-28 new)
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 21: Client-Server Payment Gateway (Paillier + RSA Sign + SHA256) ---
# PASTE FULL CODE for PaymentGatewaySystem Class and menu_payment_gateway() from V5 HERE
# (Using placeholder from V5)
def menu_payment_gateway(): # System 21
    if HAS_PAILLIER and HAS_CRYPTO: print("\n[Placeholder Sys 21: Payment Gateway]\nPASTE CODE HERE"); pause()
    else: print("\nSys 21 requires 'phe' and 'pycryptodome'."); pause()

# --- System 22: Secure Aggregation (Paillier + ElGamal Sign + SHA-512) ---
# PASTE FULL CODE for SecureAggregationPaillierElGamal Class and menu function from V5 HERE
# (Using placeholder from V5)
def menu_secure_aggregation_paillier_elgamal(): # System 22
    if HAS_PAILLIER and HAS_CRYPTO: print("\n[Placeholder Sys 22: Secure Aggregation (P+ElG)]\nPASTE CODE HERE"); pause()
    else: print("\nSys 22 requires 'phe' and 'pycryptodome'."); pause()

# --- System 23: Homomorphic Product Demo (ElGamal Enc + RSA Sign + SHA256) ---
# PASTE FULL CODE for HomomorphicProductElGamal Class and menu function from V5 HERE
# (Using placeholder from V5)
def menu_homomorphic_product_elgamal(): # System 23
    if HAS_CRYPTO: print("\n[Placeholder Sys 23: Homomorphic Product (ElG+RSA)]\nPASTE CODE HERE"); pause()
    else: print("\nSys 23 requires 'pycryptodome'."); pause()

# --- System 24: Secure Aggregation (Paillier + RSA Sign + SHA-512) ---
# PASTE FULL CODE for SecureAggregationPaillierRSA Class and menu function from V5 HERE
# (Using placeholder from V5)
def menu_secure_aggregation_paillier_rsa(): # System 24
    if HAS_PAILLIER and HAS_CRYPTO: print("\n[Placeholder Sys 24: Secure Aggregation (P+RSA)]\nPASTE CODE HERE"); pause()
    else: print("\nSys 24 requires 'phe' and 'pycryptodome'."); pause()

# --- System 25: Secure Transmission (AES-GCM + ElGamal Sign + SHA256) ---
# PASTE FULL CODE for SecureTransmissionAESElGamal Class and menu function from V5 HERE
# (Using placeholder from V5)
def menu_secure_transmission_aes_elgamal(): # System 25
    if HAS_CRYPTO: print("\n[Placeholder Sys 25: Secure Tx (AES+ElG)]\nPASTE CODE HERE"); pause()
    else: print("\nSys 25 requires 'pycryptodome'."); pause()

# --- System 26: Secure Storage (Rabin Enc + RSA Sign + SHA512) ---
# PASTE FULL CODE for SecureStorageRabinRSA Class and menu function from V5 HERE
# (Using placeholder from V5)
def menu_secure_storage_rabin_rsa(): # System 26
    if HAS_CRYPTO: print("\n[Placeholder Sys 26: Secure Storage (Rabin+RSA)]\nPASTE CODE HERE"); pause()
    else: print("\nSys 26 requires 'pycryptodome'."); pause()


# --- System 27 (NEW): Signed Encryption (RSA Encrypt + ElGamal Sign + SHA1) ---
if HAS_CRYPTO:
    class SignedEncryptionRSAElGamal:
        def __init__(self, rsa_bits=1024, elgamal_bits=1024):
            self.performance = PerformanceTracker()
            print(f"\nInitializing Signed Encryption (RSA {rsa_bits} Enc, ElGamal {elgamal_bits} Sig)...")
            start=time.time(); self.rsa_key = RSA.generate(rsa_bits); self.rsa_pub_key = self.rsa_key.publickey(); self.performance.record('RSA_KeyGen_Enc', time.time()-start); print("  ✓ RSA encryption keys.")
            start=time.time(); self.elgamal_keys = generate_elgamal_sig_keys(bits=elgamal_bits); self.performance.record('ElGamal_KeyGen_Sig', time.time()-start); print("  ✓ ElGamal signature keys.")
            self.messages = [] # {'id', 'enc_data', 'signature', 'hash_hex'}

        def send_message(self, msg_id, message):
            print(f"\nSending message {msg_id}...")
            msg_bytes = message.encode('utf-8')

            # Sign hash of original message with ElGamal
            start=time.time(); hash_obj = SHA1.new(msg_bytes); hash_val = hash_obj.digest(); self.performance.record('SHA1_Hash', time.time()-start)
            start=time.time(); signature = elgamal_sign(hash_val, self.elgamal_keys); self.performance.record('ElGamal_Sign', time.time()-start)

            # Encrypt original message with RSA Public key
            start=time.time(); cipher_rsa = PKCS1_OAEP.new(self.rsa_pub_key); enc_data = cipher_rsa.encrypt(msg_bytes); self.performance.record('RSA_Encrypt', time.time()-start, len(msg_bytes))

            record = {'id': msg_id, 'enc_data': enc_data, 'signature': signature, 'hash_hex': hash_obj.hexdigest()}
            self.messages.append(record)
            print("  ✓ Message encrypted and signed.")
            return record

        def receive_message(self, record):
            print(f"\nReceiving message {record['id']}...")
            elg_pub_key = {k:v for k,v in self.elgamal_keys.items() if k != 'x'}

            # Decrypt message first using RSA Private key
            start=time.time(); cipher_rsa = PKCS1_OAEP.new(self.rsa_key)
            try: dec_bytes = cipher_rsa.decrypt(record['enc_data']); dec_msg = dec_bytes.decode('utf-8'); self.performance.record('RSA_Decrypt', time.time()-start)
            except (ValueError, TypeError) as e: print(f"  ❌ Decryption failed: {e}"); self.performance.record('RSA_Decrypt', time.time()-start); return None, False

            print(f"  ✓ Decrypted: '{dec_msg}'")

            # Verify signature against decrypted message
            start=time.time(); hash_recomputed = SHA1.new(dec_bytes).digest(); is_valid = elgamal_verify(hash_recomputed, record['signature'], elg_pub_key); self.performance.record('ElGamal_Verify', time.time()-start)
            print(f"  ✓ Signature Verification: {'VALID' if is_valid else 'INVALID'}")
            # Compare hashes just for info
            print(f"    Received Hash (SHA1): {record['hash_hex']}")
            print(f"    Computed Hash (SHA1): {SHA1.new(dec_bytes).hexdigest()}")

            return dec_msg, is_valid

    def menu_signed_encryption_rsa_elgamal(): # System 27
        if not HAS_CRYPTO: print("\nSystem 27 requires 'pycryptodome'."); pause(); return
        system = SignedEncryptionRSAElGamal(); msg_count = 0; last_msg_record = None
        while True:
            print(f"\n{'='*70}\n  ✍️ SYSTEM 27: SIGNED ENCRYPTION (RSA Enc + ElGamal Sig + SHA1)\n{'='*70}")
            print("  1. Send Message | 2. Receive Last Message | 3. View Last Record | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': msg_count+=1; msg = input("Message to send: "); last_msg_record = system.send_message(f"M{msg_count}", msg)
            elif choice == '2':
                if last_msg_record: msg, valid = system.receive_message(last_msg_record);
                    if msg is not None: print(f"\n  Received Content: {msg} (Signature Valid: {valid})")
                else: print("No message sent yet.")
            elif choice == '3':
                if last_msg_record: print(f"\nRecord ID: {last_msg_record['id']}\n Enc: {last_msg_record['enc_data'].hex()[:40]}...\n Sig: {last_msg_record['signature']}\n Hash: {last_msg_record['hash_hex']}")
                else: print("No record.")
            elif choice == '4': system.performance.print_graph("Signed Encryption Performance")
            elif choice == '5': system.performance.compare_algorithms(['RSA_Encrypt', 'ElGamal_Sign', 'SHA1_Hash'])
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_signed_encryption_rsa_elgamal(): print("\nSystem 27 requires 'pycryptodome'."); pause()


# --- System 28 (NEW): Encrypt-then-MAC (AES-CBC + HMAC-SHA256) ---
if HAS_CRYPTO:
    from Crypto.Hash import HMAC # Import HMAC specifically

    class EncryptThenMAC:
        def __init__(self, aes_key_size=32, mac_key_size=32): # AES-256, SHA256-HMAC
            self.performance = PerformanceTracker()
            print(f"\nInitializing Encrypt-then-MAC (AES-{aes_key_size*8} CBC, HMAC-SHA256)...")
            start=time.time(); self.aes_key = get_random_bytes(aes_key_size); self.performance.record('AES_KeyGen', time.time()-start)
            start=time.time(); self.mac_key = get_random_bytes(mac_key_size); self.performance.record('HMAC_KeyGen', time.time()-start) # Separate key for MAC is crucial
            self.messages = [] # {'id', 'iv', 'ciphertext', 'mac_tag'}

        def protect_message(self, msg_id, message):
            print(f"\nProtecting message {msg_id}...")
            msg_bytes = message.encode('utf-8')

            # Encrypt with AES-CBC
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_CBC); iv = cipher_aes.iv; ciphertext = cipher_aes.encrypt(pad(msg_bytes, AES.block_size)); self.performance.record('AES_Encrypt_CBC', time.time()-start, len(msg_bytes))

            # Compute HMAC-SHA256 over IV + Ciphertext
            start=time.time(); hmac_sha256 = HMAC.new(self.mac_key, digestmod=SHA256); hmac_sha256.update(iv + ciphertext); mac_tag = hmac_sha256.digest(); self.performance.record('HMAC_SHA256_Generate', time.time()-start)

            record = {'id': msg_id, 'iv': iv, 'ciphertext': ciphertext, 'mac_tag': mac_tag}
            self.messages.append(record)
            print("  ✓ Message encrypted and MAC generated.")
            return record

        def verify_and_decrypt(self, record):
            print(f"\nVerifying and decrypting message {record['id']}...")

            # Verify HMAC first
            start=time.time(); hmac_sha256 = HMAC.new(self.mac_key, digestmod=SHA256); hmac_sha256.update(record['iv'] + record['ciphertext']);
            try: hmac_sha256.verify(record['mac_tag']); is_valid = True; self.performance.record('HMAC_SHA256_Verify', time.time()-start)
            except ValueError: is_valid = False; self.performance.record('HMAC_SHA256_Verify', time.time()-start); print("  ❌ INVALID MAC TAG! Message may be tampered."); return None

            print("  ✓ MAC Tag VALID.")

            # Decrypt using AES-CBC
            start=time.time(); cipher_aes = AES.new(self.aes_key, AES.MODE_CBC, iv=record['iv'])
            try: dec_bytes = unpad(cipher_aes.decrypt(record['ciphertext']), AES.block_size); dec_msg = dec_bytes.decode('utf-8'); self.performance.record('AES_Decrypt_CBC', time.time()-start)
            except (ValueError, KeyError) as e: print(f"  ❌ Decryption failed (padding error?): {e}"); self.performance.record('AES_Decrypt_CBC', time.time()-start); return None

            print(f"  ✓ Decrypted successfully: '{dec_msg}'")
            return dec_msg

    def menu_encrypt_then_mac(): # System 28
        if not HAS_CRYPTO: print("\nSystem 28 requires 'pycryptodome'."); pause(); return
        system = EncryptThenMAC(); msg_count = 0; last_msg_record = None
        while True:
            print(f"\n{'='*70}\n  🔐 SYSTEM 28: ENCRYPT-THEN-MAC (AES-CBC + HMAC-SHA256)\n{'='*70}")
            print("  1. Protect Message | 2. Verify & Decrypt Last | 3. View Last Record | 4. Performance | 5. Compare Algs | 6. Back"); print("-"*70)
            choice = input("Choice: ").strip()
            if choice == '1': msg_count+=1; msg = input("Message: "); last_msg_record = system.protect_message(f"EtM_{msg_count}", msg)
            elif choice == '2':
                if last_msg_record: msg = system.verify_and_decrypt(last_msg_record);
                    if msg is not None: print(f"\n  Verified Content: {msg}")
                else: print("No message protected yet.")
            elif choice == '3':
                if last_msg_record: print(f"\nRecord ID: {last_msg_record['id']}\n IV: {last_msg_record['iv'].hex()}\n CT: {last_msg_record['ciphertext'].hex()[:40]}...\n MAC: {last_msg_record['mac_tag'].hex()}")
                else: print("No record.")
            elif choice == '4': system.performance.print_graph("Encrypt-then-MAC Performance")
            elif choice == '5': system.performance.compare_algorithms(['AES_Encrypt_CBC', 'HMAC_SHA256_Generate', 'HMAC_SHA256_Verify']) # Compare AES Enc vs HMAC Gen vs HMAC Verify
            elif choice == '6': break
            else: print("Invalid choice.")
            if choice != '6': pause()

else:
    def menu_encrypt_then_mac(): print("\nSystem 28 requires 'pycryptodome'."); pause()


# ═══════════════════════════════════════════════════════════════════════════
#
# PART 4: ADVANCED CONCEPTS (PLACEHOLDERS)
# Systems 29-30
#
# ═══════════════════════════════════════════════════════════════════════════

def menu_sse_placeholder(): # System 29
    clear_screen()
    print("\n--- SYSTEM 29: Symmetric Searchable Encryption (SSE) ---")
    print("CONCEPT: Allows searching over encrypted data using symmetric keys.")
    print("USE CASES: Secure cloud storage search, encrypted databases.")
    print("IMPLEMENTATION: Complex, involves encrypted indexes/trapdoors. Not implemented here.")
    print("CHALLENGES: Security against leakage (access patterns, search patterns).")
    pause()

def menu_pkse_placeholder(): # System 30
    clear_screen()
    print("\n--- SYSTEM 30: Public Key Searchable Encryption (PKSE) ---")
    print("CONCEPT: Allows searching over data encrypted with a public key, often by a third party.")
    print("USE CASES: Secure email filtering, outsourced secure databases.")
    print("IMPLEMENTATION: Very complex, often uses bilinear pairings or other advanced crypto. Not implemented here.")
    print("CHALLENGES: Security against leakage, performance overhead.")
    pause()

# ═══════════════════════════════════════════════════════════════════════════
#
# PART 5: TOOLS & EXIT (Renumbered)
# Systems 31+
#
# ═══════════════════════════════════════════════════════════════════════════

# --- System 31: Universal Benchmark ---
# PASTE FULL CODE for run_comprehensive_benchmark() from V5 HERE
# (Using placeholder from V5)
def run_comprehensive_benchmark(): # System 31
    print("\n[Placeholder Sys 31: Universal Benchmark]\nPASTE CODE HERE"); pause()


# --- Master Menu ---
def master_menu():
    """Master menu to select systems"""
    while True:
        clear_screen()
        print("\n" + "="*70)
        print("  💻 ICT3141 ULTIMATE EXAM SCRIPT V6 - MASTER MENU")
        print("="*70)

        # List all parts and systems concisely
        print("\n--- PART 1: BASE 3-ALGORITHM SYSTEMS (1-12) ---")
        # (Assume user knows these from previous script)
        print("  [Includes Secure Email, Banking, Cloud, Legacy, Healthcare, Docs, Msg, Transfer, Library, Chat, E-Voting, Hybrid]")

        print("\n--- PART 2: INTERACTIVE CLASSICAL CIPHERS (13-20) ---")
        print("  13. Additive | 14. Multiplicative | 15. Affine | 16. Vigenère")
        print("  17. Autokey  | 18. Playfair       | 19. Hill   | 20. Transposition")

        print("\n--- PART 3: EXAM-SPECIFIC & NEW COMBINATIONS (21-28) ---")
        print("  21. Payment Gateway (Paillier+RSA+SHA256)")
        print("  22. Secure Aggregation (Paillier+ElG+SHA512)")
        print("  23. Homomorphic Product (ElG Enc+RSA+SHA256)")
        print("  24. Secure Aggregation (Paillier+RSA+SHA512)")
        print("  25. Secure Tx (AES-GCM+ElG+SHA256)")
        print("  26. Secure Storage (Rabin+RSA+SHA512)")
        print("  27. Signed Encryption (RSA Enc+ElG+SHA1)")
        print("  28. Encrypt-then-MAC (AES-CBC+HMAC-SHA256)")

        print("\n--- PART 4: ADVANCED CONCEPTS (Placeholders) (29-30) ---")
        print("  29. SSE Conceptual   | 30. PKSE Conceptual")

        print("\n--- PART 5: TOOLS & EXIT (31-33) ---")
        print("  31. Universal Algorithm Benchmark")
        print("  32. About This Script")
        print("  33. Exit")
        print("-"*70)

        choice = input("\nSelect system (1-33): ").strip()

        # Map choices to functions
        menu_map = {
            '1': menu_email_system, '2': menu_banking_system, '3': menu_cloud_system, '4': menu_legacy_banking,
            '5': menu_healthcare, '6': menu_document_management, '7': menu_messaging, '8': menu_file_transfer,
            '9': menu_digital_library, '10': menu_secure_chat, '11': menu_voting_system, '12': menu_hill_hybrid,
            # Part 2
            '13': menu_additive_cipher, '14': menu_multiplicative_cipher, '15': menu_affine_cipher, '16': menu_vigenere_cipher,
            '17': menu_autokey_cipher, '18': menu_playfair_cipher, '19': menu_hill_cipher, '20': menu_transposition_cipher,
            # Part 3
            '21': menu_payment_gateway, '22': menu_secure_aggregation_paillier_elgamal, '23': menu_homomorphic_product_elgamal,
            '24': menu_secure_aggregation_paillier_rsa, '25': menu_secure_transmission_aes_elgamal, '26': menu_secure_storage_rabin_rsa,
            '27': menu_signed_encryption_rsa_elgamal, '28': menu_encrypt_then_mac,
            # Part 4
            '29': menu_sse_placeholder, '30': menu_pkse_placeholder,
            # Part 5
            '31': run_comprehensive_benchmark
        }

        if choice in menu_map:
             # Simplified check - assumes function name indicates dependency
             func = menu_map[choice]
             func_name = func.__name__
             disabled = False
             if ("hill" in func_name or "affine_playfair" in func_name) and not HAS_NUMPY: disabled = True; lib = "NumPy"
             elif ("paillier" in func_name or "voting" in func_name or "gateway" in func_name or "aggregation" in func_name) and not HAS_PAILLIER: disabled = True; lib = "'phe' (Paillier)"
             elif ("elgamal" in func_name or "rsa" in func_name or "aes" in func_name or "des" in func_name or "rabin" in func_name or "mac" in func_name) and not HAS_CRYPTO: disabled = True; lib = "PyCryptodome"

             if disabled: print(f"\nSystem {choice} requires {lib} (not found)."); pause()
             else: func() # Call the function
        elif choice == '32': # About
             clear_screen(); print("\n" + "="*70 + "\n  ABOUT THIS SCRIPT\n" + "="*70 + "\n  ICT3141 Ultimate Exam Script V6\n  Includes systems 1-28, placeholders 29-30, benchmark 31.\n" + "="*70); pause()
        elif choice == '33': # Exit
             print("\nGood luck! 🍀"); break
        else: print("Invalid choice."); pause()

# --- Main Execution Block ---
if __name__ == "__main__":
    # (Same try/except block as V5)
    try: master_menu()
    except KeyboardInterrupt: print("\n\nExiting."); sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}\n"); import traceback; traceback.print_exc()
        print("\nEnsure libraries (pycryptodome, numpy, phe, matplotlib) are installed."); sys.exit(1)
