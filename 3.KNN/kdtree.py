import numpy as np


class Node:

    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def median(arr):
    m = int(len(arr) // 2)
    arr = np.sort(arr)
    return arr[m]


def build_kdtree(data, d):
    if len(data) == 0:
        return
    k = len(data[0])
    l = d % k
    medi = median(data[:, l])
    median_index = np.where(data[:, l] == medi)  #[0][0]
    tree = Node(data[median_index])
    data = np.delete(data, median_index, axis=0)
    data_left = data[data[:, l] <= medi]
    data_right = data[data[:, l] > medi]
    tree.left = build_kdtree(data_left, d + 1)
    tree.right = build_kdtree(data_right, d + 1)
    return tree


def BFS(root):
    queue = [root]
    while queue:
        n = len(queue)
        for i in range(n):
            q = queue.pop(0)
            if q:
                print(q.value)
                queue.append(q.left if q.left else None)
                queue.append(q.right if q.right else None)


if __name__ == "__main__":
    a = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    tree = build_kdtree(a, 0)
    BFS(tree)
