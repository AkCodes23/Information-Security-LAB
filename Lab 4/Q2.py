import os, json, secrets, base64, logging
from datetime import datetime, timedelta
from Crypto.Util import number
from Crypto.Cipher import AES

# === Config ===
STORE_DIR = "./rabin_kms_store"
os.makedirs(STORE_DIR, exist_ok=True)

# Master key for encrypting private keys at rest
MASTER_KEY_B64 = os.environ.get("MASTER_KEY") or base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
MASTER_KEY = base64.urlsafe_b64decode(MASTER_KEY_B64)

VALIDITY_DAYS = 365

# === Logging ===
logging.basicConfig(filename="rabin_kms_audit.log",
                    level=logging.INFO,
                    format="%(asctime)s %(message)s")

def log_event(event, facility, **extra):
    logging.info(json.dumps({"event": event, "facility": facility, **extra}))

# === Rabin Keygen ===
def blum_prime(bits):
    while True:
        p = number.getPrime(bits)
        if p % 4 == 3:
            return p

def rabin_keypair(bits=1024):
    p = blum_prime(bits // 2)
    q = blum_prime(bits // 2)
    return {"n": p * q, "p": p, "q": q, "bits": bits}

# === AES-GCM Encrypt/Decrypt ===
def enc_private_key(priv, facility):
    aad = f"facility:{facility}".encode()
    nonce = secrets.token_bytes(12)
    cipher = AES.new(MASTER_KEY, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ct, tag = cipher.encrypt_and_digest(json.dumps(priv).encode())
    return {"nonce": nonce.hex(), "ct": ct.hex(), "tag": tag.hex()}

def dec_private_key(blob, facility):
    aad = f"facility:{facility}".encode()
    cipher = AES.new(MASTER_KEY, AES.MODE_GCM, nonce=bytes.fromhex(blob["nonce"]))
    cipher.update(aad)
    pt = cipher.decrypt_and_verify(bytes.fromhex(blob["ct"]), bytes.fromhex(blob["tag"]))
    return json.loads(pt.decode())

# === Storage ===
def record_path(fid): return os.path.join(STORE_DIR, f"{fid}.json")
def save_record(rec): open(record_path(rec["facility"]), "w").write(json.dumps(rec))
def load_record(fid): return json.loads(open(record_path(fid)).read()) if os.path.exists(record_path(fid)) else None

# === Core operations ===
def register_facility(fid, bits=1024):
    if load_record(fid): raise ValueError("Facility exists")
    keys = rabin_keypair(bits)
    expires = (datetime.utcnow() + timedelta(days=VALIDITY_DAYS)).isoformat() + "Z"
    rec = {
        "facility": fid,
        "public": {"n": str(keys["n"]), "bits": bits},
        "private_enc": enc_private_key({"p": keys["p"], "q": keys["q"]}, fid),
        "expires": expires,
        "revoked": False
    }
    save_record(rec)
    log_event("register", fid, bits=bits)
    return rec

def get_public(fid):
    rec = load_record(fid)
    return None if not rec or rec["revoked"] else rec["public"]

def retrieve_private(fid):
    rec = load_record(fid)
    return None if not rec or rec["revoked"] else dec_private_key(rec["private_enc"], fid)

def revoke_facility(fid):
    rec = load_record(fid)
    if not rec: return False
    rec["revoked"] = True
    save_record(rec)
    log_event("revoke", fid)
    return True

def renew_facility(fid):
    rec = load_record(fid)
    if not rec or rec["revoked"]: return False
    keys = rabin_keypair(rec["public"]["bits"])
    rec["public"] = {"n": str(keys["n"]), "bits": rec["public"]["bits"]}
    rec["private_enc"] = enc_private_key({"p": keys["p"], "q": keys["q"]}, fid)
    rec["expires"] = (datetime.utcnow() + timedelta(days=VALIDITY_DAYS)).isoformat() + "Z"
    save_record(rec)
    log_event("renew", fid)
    return True

# === Demo ===
if __name__ == "__main__":
    print("MASTER_KEY (save this):", MASTER_KEY_B64)

    reg = register_facility("clinicA", 1024)
    print("Registered:", reg["public"])

    pub = get_public("clinicA")
    print("Public key:", pub)

    priv = retrieve_private("clinicA")
    print("Private key parts:", priv)

    renew_facility("clinicA")
    print("Renewed public:", get_public("clinicA"))

    revoke_facility("clinicA")
    print("After revoke:", get_public("clinicA"))
