def find(n, k):
    quotient = k // (n - 1)
    remainder = k % (n - 1)
    if remainder == 0:
        return n * quotient - 1
    else:
        return n * quotient + remainder

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        result = find(n, k)
        print(result)

if __name__ == '__main__':
    main()
