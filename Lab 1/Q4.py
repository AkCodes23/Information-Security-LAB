import numpy as np

def preprocess(text):
    text = text.lower().replace(" ", "").replace("j", "i")
    if len(text) % 2 != 0:
        text += 'x'
    return text

def hill_encrypt(text, key_matrix):
    text = preprocess(text)
    encrypted = ''
    for i in range(0, len(text), 2):
        pair = [ord(text[i]) - ord('a'), ord(text[i+1]) - ord('a')]
        res = np.dot(key_matrix, pair) % 26
        encrypted += chr(res[0] + ord('a')) + chr(res[1] + ord('a'))
    return encrypted

def main():
    message = "We live in an insecure world"
    key_matrix = np.array([[3, 3], [2, 7]])
    cipher = hill_encrypt(message, key_matrix)
    print("Hill Cipher → Encrypted:", cipher)

if __name__ == "__main__":
    main()
