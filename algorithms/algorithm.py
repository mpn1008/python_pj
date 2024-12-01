# A sorted list of 150 numbers
# import math
#
# numbers = [
#     1, 2, 4, 5, 6, 7, 8, 9, 10,
#     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#     31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#     51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
#     61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
#     71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
#     81, 82, 83, 84, 85, 87, 88, 89, 90,
#     91, 92, 93, 94, 95, 96, 97, 98, 99, 103, 104, 105, 106, 107, 108, 109, 110,
#     111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
#     121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
#     131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
#     141, 142, 143, 144, 145, 146, 147, 148, 149, 150
# ]
#
# # Print the sorted list
# print(numbers)
#
#
# def binary_search(number, data):
#     low = 0
#     high = len(data) - 1
#
#     while low <= high:
#         return None
#     return None

# def binary_search(list, item):
#     low = 0
#     high = len(list)-1
#     cnt = 0
#     while low <= high:
#         cnt+=1
#         mid = int((math.floor(low + high)) / 2)
#         guess = list[mid]
#         print(f"guess:{guess} , cnt: {cnt}")
#         if guess == item:
#             return mid
#         if guess > item:
#             high = mid - 1
#         else:
#             low = mid + 1
#     return None
#
# print(binary_search(numbers, 150))

from showcallstack import showcallstack
def fibo(n, mem = None):

    if mem is None:
        mem = {}

    if n <= 1:
        return n

    if n in mem:
        return mem[n]

    mem[n] = (fibo(n - 1, mem) + fibo(n - 2, mem))
    return mem[n]

print(fibo(10))

# def fibo(s, n):
#     print(s)
#     print(n)
#     if n <= 1:
#         return n
#     else:
#         return fibo("-1 flow_______", n-1) + fibo("-2 flow______", n-2)
#
#
# print(fibo("start", 10))