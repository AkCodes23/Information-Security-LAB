
import sys
from datetime import datetime
from Crypto.Util import number
from Crypto.Hash import SHA512, SHA256

# --------------- Simple Rabin Cryptosystem (Blum integers) ---------------

def generate_rabin_keys(bits=2048):
    # p, q must be primes where p % 4 == q % 4 == 3
    while True:
        p = number.getPrime(bits // 2)
        if p % 4 == 3:
            break
    while True:
        q = number.getPrime(bits // 2)
        if q % 4 == 3 and q != p:
            break
    n = p * q
    return {"n": n, "p": p, "q": q}

def _rabin_encode(msg_bytes):
    # Add minimal redundancy to identify the correct root after decryption
    # Format: b"RB" + 2-byte length + msg + SHA256(msg)[:16]
    if not isinstance(msg_bytes, (bytes, bytearray)):
        raise ValueError("msg_bytes must be bytes")
    L = len(msg_bytes)
    if L > 65535:
        raise ValueError("Message too long for simple encoding")
    length_bytes = L.to_bytes(2, byteorder="big")
    tag = SHA256.new(msg_bytes).digest()[:16]
    return b"RB" + length_bytes + msg_bytes + tag

def rabin_encrypt(msg_bytes, n):
    enc = _rabin_encode(msg_bytes)
    m = number.bytes_to_long(enc)
    if m >= n:
        raise ValueError("Message too long for Rabin modulus. Use shorter message or larger key.")
    c = pow(m, 2, n)
    return c

def rabin_decrypt(cipher_int, p, q, n):
    # Compute square roots modulo p and q (p,q mod 4 == 3)
    mp = pow(cipher_int, (p + 1) // 4, p)
    mq = pow(cipher_int, (q + 1) // 4, q)

    # CRT Combine to get four roots
    inv_p_mod_q = number.inverse(p, q)
    inv_q_mod_p = number.inverse(q, p)
    a = (q * inv_q_mod_p) % n
    b = (p * inv_p_mod_q) % n

    roots = []
    roots.append((a * mp + b * mq) % n)
    roots.append((a * mp - b * mq) % n)
    roots.append((-a * mp + b * mq) % n)
    roots.append((-a * mp - b * mq) % n)

    # Try to decode redundancy
    for r in roots:
        rb = number.long_to_bytes(r)
        if len(rb) < 20:
            continue
        if not (rb[0:2] == b"RB"):
            continue
        if len(rb) < 2 + 2 + 16:
            continue
        L = int.from_bytes(rb[2:4], byteorder="big")
        if L < 0 or L > 65535:
            continue
        if len(rb) != 2 + 2 + L + 16:
            continue
        msg = rb[4:4 + L]
        tag = rb[4 + L:]
        if SHA256.new(msg).digest()[:16] == tag:
            return msg
    return None

# --------------- ElGamal Signatures ---------------

def generate_safe_prime(bits):
    # Try to generate a safe prime p=2q+1
    # This can take a few attempts for larger bit sizes
    while True:
        q = number.getPrime(bits - 1)
        p = 2 * q + 1
        if number.isPrime(p):
            return p, q

def find_generator_for_safe_prime(p, q):
    # For safe prime p = 2q + 1, a generator g must satisfy:
    # g^2 mod p != 1 and g^q mod p != 1
    while True:
        g = number.getRandomRange(2, p - 2)
        if pow(g, 2, p) != 1 and pow(g, q, p) != 1:
            return g

def generate_elgamal_keys(bits=1024):
    p, q = generate_safe_prime(bits)
    g = find_generator_for_safe_prime(p, q)
    x = number.getRandomRange(2, p - 2)
    y = pow(g, x, p)
    return {"p": p, "g": g, "x": x, "y": y}

def _elg_hash_to_int(msg_bytes, p_minus_1):
    h = SHA512.new(msg_bytes).digest()
    h_int = number.bytes_to_long(h) % p_minus_1
    if h_int == 0:
        h_int = 1
    return h_int

def elgamal_sign(msg_bytes, priv):
    p = priv["p"]
    g = priv["g"]
    x = priv["x"]
    k = None
    while True:
        k = number.getRandomRange(2, p - 2)
        if number.GCD(k, p - 1) == 1:
            break
    r = pow(g, k, p)
    kinv = number.inverse(k, p - 1)
    h_int = _elg_hash_to_int(msg_bytes, p - 1)
    s = (kinv * (h_int - x * r)) % (p - 1)
    return (int(r), int(s))

def elgamal_verify(msg_bytes, sig, pub):
    p = pub["p"]
    g = pub["g"]
    y = pub["y"]
    r, s = sig
    if not (1 <= r <= p - 1):
        return False
    if not (0 <= s <= p - 2):
        return False
    h_int = _elg_hash_to_int(msg_bytes, p - 1)
    v1 = (pow(y, r, p) * pow(r, s, p)) % p
    v2 = pow(g, h_int, p)
    return v1 == v2

# --------------- In-Memory Store and Utilities ---------------

class Store:
    def __init__(self):
        self.keys = {
            "rabin": None,     # {"n":..., "p":..., "q":...}
            "elgamal": None,   # {"p":..., "g":..., "x":..., "y":...}
        }
        self.customer_history = []      # list of transactions sent by customer
        self.merchant_inbox = []        # list of pending transactions for merchant
        self.merchant_processed = []    # list of processed transactions
        self.next_tx_id = 1

store = Store()

def now_ts():
    return datetime.now().isoformat(timespec="seconds")

def short_hex(x, length=32):
    hx = hex(x)[2:]
    if len(hx) <= length:
        return hx
    return hx[:length] + "...(" + str(len(hx)) + " hex chars)"

# --------------- Role Actions ---------------

def customer_create_and_send():
    if store.keys["rabin"] is None or store.keys["elgamal"] is None:
        print("Keys not ready.")
        return

    try:
        details = input("Enter payment details (example: Send 55000 to Bob using Mastercard 3048330330393783): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not details:
        print("Empty details. Aborting.")
        return

    msg_bytes = details.encode("utf-8")
    ts = now_ts()
    # Hash (SHA-512) of plaintext; this is what we sign and send as integrity record
    digest = SHA512.new(msg_bytes).digest()
    digest_hex = digest.hex()

    # Sign the digest using ElGamal (signing input is digest bytes)
    sig_r, sig_s = elgamal_sign(digest, store.keys["elgamal"])

    # Encrypt payment details using Rabin (merchant's public n)
    n = store.keys["rabin"]["n"]
    try:
        c = rabin_encrypt(msg_bytes, n)
    except Exception as e:
        print("Encryption error:", str(e))
        return

    tx = {
        "id": store.next_tx_id,
        "timestamp": ts,
        "plaintext": details,
        "cipher_hex": hex(c)[2:],   # store as hex text for display
        "hash_hex": digest_hex,
        "sig_r": sig_r,
        "sig_s": sig_s,
        "processed": False,
    }
    store.next_tx_id += 1
    store.customer_history.append(tx)
    store.merchant_inbox.append(dict(tx))  # copy to merchant inbox

    print("Transaction created and sent to merchant.")
    print("ID:", tx["id"])
    print("Timestamp:", tx["timestamp"])
    print("Before encryption (plaintext):", tx["plaintext"])
    print("After encryption (cipher hex):", short_hex(c))
    print("SHA-512 digest (hex):", tx["hash_hex"][:64] + "..." if len(tx["hash_hex"]) > 64 else tx["hash_hex"])
    print("ElGamal signature r:", tx["sig_r"])
    print("ElGamal signature s:", tx["sig_s"])

def customer_view_history():
    if not store.customer_history:
        print("No past transactions.")
        return
    for tx in store.customer_history:
        print("ID:", tx["id"])
        print("Timestamp:", tx["timestamp"])
        print("Plaintext:", tx["plaintext"])
        print("Cipher (hex):", tx["cipher_hex"])
        print("SHA-512 digest (hex):", tx["hash_hex"])
        print("Signature r:", tx["sig_r"])
        print("Signature s:", tx["sig_s"])
        print("Processed by merchant:", tx["processed"])
        print("-" * 40)

def merchant_process_all():
    if store.keys["rabin"] is None or store.keys["elgamal"] is None:
        print("Keys not ready.")
        return
    if not store.merchant_inbox:
        print("No pending transactions.")
        return

    p = store.keys["rabin"]["p"]
    q = store.keys["rabin"]["q"]
    n = store.keys["rabin"]["n"]

    pub_elg = {
        "p": store.keys["elgamal"]["p"],
        "g": store.keys["elgamal"]["g"],
        "y": store.keys["elgamal"]["y"],
    }

    processed_any = False
    new_inbox = []
    for tx in store.merchant_inbox:
        processed_any = True
        ts = now_ts()
        c_hex = tx["cipher_hex"]
        try:
            c = int(c_hex, 16)
        except Exception:
            c = None

        decrypted_msg = None
        decrypt_ok = False
        if c is not None:
            try:
                m = rabin_decrypt(c, p, q, n)
                if m is not None:
                    decrypted_msg = m.decode("utf-8", errors="replace")
                    decrypt_ok = True
                else:
                    decrypt_ok = False
            except Exception:
                decrypt_ok = False

        # Compute hash of decrypted plaintext
        computed_hash_hex = "DECRYPT_FAIL"
        if decrypt_ok:
            computed_hash_hex = SHA512.new(m).hexdigest()

        # Verify signature using received hash bytes (auditable)
        received_hash_hex = tx["hash_hex"]
        try:
            received_hash_bytes = bytes.fromhex(received_hash_hex)
        except Exception:
            received_hash_bytes = b""
        sig = (tx["sig_r"], tx["sig_s"])
        sig_valid = elgamal_verify(received_hash_bytes, sig, pub_elg)

        # Optionally also verify that signature corresponds to computed hash if decryption ok
        hash_match = False
        if decrypt_ok:
            hash_match = (computed_hash_hex == received_hash_hex)

        rec = {
            "id": tx["id"],
            "timestamp": ts,
            "received_hash_hex": received_hash_hex,
            "computed_hash_hex": computed_hash_hex,
            "signature_valid": bool(sig_valid),
            "hash_match": bool(hash_match),
            "decryption_ok": bool(decrypt_ok),
            "decrypted_plaintext": decrypted_msg if decrypt_ok else "",
        }
        store.merchant_processed.append(rec)
        tx["processed"] = True

        print("Processed transaction ID:", tx["id"])
        print("Signature valid:", rec["signature_valid"])
        print("Decryption ok:", rec["decryption_ok"])
        print("Received hash (hex):", rec["received_hash_hex"])
        print("Computed hash (hex):", rec["computed_hash_hex"])
        print("Hashes match:", rec["hash_match"])
        print("Timestamp:", rec["timestamp"])
        print("-" * 40)

    # Clear inbox after processing all
    store.merchant_inbox = []
    if not processed_any:
        print("No transactions processed.")

def merchant_show_processed():
    if not store.merchant_processed:
        print("No processed records.")
        return
    for rec in store.merchant_processed:
        print("ID:", rec["id"])
        print("Timestamp:", rec["timestamp"])
        print("Signature valid:", rec["signature_valid"])
        print("Decryption ok:", rec["decryption_ok"])
        print("Received hash (hex):", rec["received_hash_hex"])
        print("Computed hash (hex):", rec["computed_hash_hex"])
        print("Hashes match:", rec["hash_match"])
        # Merchant can see plaintext if needed:
        if rec["decryption_ok"]:
            print("Decrypted plaintext:", rec["decrypted_plaintext"])
        print("-" * 40)

def auditor_view_hashed_records():
    if not store.merchant_processed:
        print("No records to audit.")
        return
    for rec in store.merchant_processed:
        print("ID:", rec["id"])
        print("Timestamp:", rec["timestamp"])
        print("Received hash (hex):", rec["received_hash_hex"])
        print("Computed hash (hex):", rec["computed_hash_hex"])
        print("Hashes match:", rec["hash_match"])
        print("-" * 40)

def auditor_verify_signatures():
    if not store.merchant_processed:
        print("No records to verify.")
        return
    pub_elg = {
        "p": store.keys["elgamal"]["p"],
        "g": store.keys["elgamal"]["g"],
        "y": store.keys["elgamal"]["y"],
    }
    for rec in store.merchant_processed:
        tx_id = rec["id"]
        sig_r = None
        sig_s = None
        hash_hex = rec["received_hash_hex"]
        for tx in store.customer_history:
            if tx["id"] == tx_id:
                sig_r = tx["sig_r"]
                sig_s = tx["sig_s"]
                break
        if sig_r is None:
            print("ID:", tx_id, "Signature not found.")
            continue
        try:
            msg_bytes = bytes.fromhex(hash_hex)
        except Exception:
            msg_bytes = b""
        ok = elgamal_verify(msg_bytes, (sig_r, sig_s), pub_elg)
        print("ID:", tx_id, "Signature valid:", bool(ok))

def show_public_keys():
    if store.keys["rabin"] is None or store.keys["elgamal"] is None:
        print("Keys not ready.")
        return
    rabin_pub_n = store.keys["rabin"]["n"]
    elg_pub = {
        "p": store.keys["elgamal"]["p"],
        "g": store.keys["elgamal"]["g"],
        "y": store.keys["elgamal"]["y"],
    }
    print("Customer ElGamal public key parameters:")
    print("p (bits):", store.keys["elgamal"]["p"].bit_length())
    print("g:", short_hex(elg_pub["g"]))
    print("y:", short_hex(elg_pub["y"]))
    print("Merchant Rabin public modulus n (bits):", rabin_pub_n.bit_length())
    print("n:", short_hex(rabin_pub_n))

def main_menu():
    while True:
        print("Select role:")
        print("1. Customer")
        print("2. Merchant")
        print("3. Auditor")
        print("4. Show public keys")
        print("5. Exit")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            customer_menu()
        elif choice == "2":
            merchant_menu()
        elif choice == "3":
            auditor_menu()
        elif choice == "4":
            show_public_keys()
        elif choice == "5":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")

def customer_menu():
    while True:
        print("Customer menu:")
        print("1. Encrypt, sign, and send payment")
        print("2. View past transactions")
        print("3. Back")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            customer_create_and_send()
        elif choice == "2":
            customer_view_history()
        elif choice == "3":
            return
        else:
            print("Invalid choice.")

def merchant_menu():
    while True:
        print("Merchant menu:")
        print("1. Process all pending transactions")
        print("2. Show processed records")
        print("3. Back")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            merchant_process_all()
        elif choice == "2":
            merchant_show_processed()
        elif choice == "3":
            return
        else:
            print("Invalid choice.")

def auditor_menu():
    while True:
        print("Auditor menu:")
        print("1. View hashed payment records")
        print("2. Verify ElGamal signatures on records")
        print("3. Back")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            auditor_view_hashed_records()
        elif choice == "2":
            auditor_verify_signatures()
        elif choice == "3":
            return
        else:
            print("Invalid choice.")

def init_keys():
    print("Generating keys. This may take a moment...")
    # Rabin: 2048-bit modulus to fit typical payment strings
    store.keys["rabin"] = generate_rabin_keys(bits=2048)
    # ElGamal: 1024-bit safe prime for demo speed
    store.keys["elgamal"] = generate_elgamal_keys(bits=1024)
    print("Keys ready.")
    print("Customer has ElGamal signing key (keeps x private, shares p,g,y).")
    print("Merchant has Rabin decryption key (keeps p,q private, shares n).")

if __name__ == "__main__":
    try:
        init_keys()
        main_menu()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
