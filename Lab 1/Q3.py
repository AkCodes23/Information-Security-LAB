import string

def prepare_message(msg):
    msg = msg.upper().replace("J", "I").replace(" ", "")
    result = []
    i = 0
    while i < len(msg):
        a = msg[i]
        b = msg[i+1] if i+1 < len(msg) else 'X'
        if a == b:
            result.extend([a, 'X'])
            i += 1
        else:
            result.extend([a, b])
            i += 2
    if len(result) % 2 != 0:
        result.append('X')
    return [result[i:i+2] for i in range(0, len(result), 2)]

def generate_playfair_matrix(key):
    key = key.upper().replace("J", "I")
    matrix = []
    seen = set()
    for char in key + string.ascii_uppercase:
        if char == 'J':
            continue
        if char not in seen:
            seen.add(char)
            matrix.append(char)
    return [matrix[i:i+5] for i in range(0, 25, 5)]

def find_coords(matrix, char):
    for row in range(5):
        for col in range(5):
            if matrix[row][col] == char:
                return row, col
    return None

def encrypt_pair(pair, matrix):
    r1, c1 = find_coords(matrix, pair[0])
    r2, c2 = find_coords(matrix, pair[1])
    if r1 == r2:
        return matrix[r1][(c1 + 1) % 5] + matrix[r2][(c2 + 1) % 5]
    elif c1 == c2:
        return matrix[(r1 + 1) % 5][c1] + matrix[(r2 + 1) % 5][c2]
    else:
        return matrix[r1][c2] + matrix[r2][c1]

def playfair_encrypt(message, keyword):
    pairs = prepare_message(message)
    matrix = generate_playfair_matrix(keyword)
    encrypted = ''.join(encrypt_pair(pair, matrix) for pair in pairs)
    return encrypted

def main():
    message = "The key is hidden under the door pad"
    keyword = "GUIDANCE"
    ciphertext = playfair_encrypt(message, keyword)
    print("Playfair Cipher → Encrypted:", ciphertext)

if __name__ == "__main__":
    main()
