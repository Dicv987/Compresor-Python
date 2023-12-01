import math

# Implementación de HeapNode y Heap para un Min-Heap
class HeapNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value

class Heap:
    def __init__(self):
        self.heap = [None]  # Inicializar con None para empezar a contar desde índice 1
        self.size = 0

    def parent(self, i):
        return i // 2

    def left_child(self, i):
        return i * 2

    def right_child(self, i):
        return i * 2 + 1

    def heapify(self, idx):
        left_idx = self.left_child(idx)
        right_idx = self.right_child(idx)
        smallest_idx = idx

        if left_idx <= self.size and self.heap[left_idx].key < self.heap[idx].key:
            smallest_idx = left_idx

        if right_idx <= self.size and self.heap[right_idx].key < self.heap[smallest_idx].key:
            smallest_idx = right_idx

        if smallest_idx != idx:
            self.heap[smallest_idx], self.heap[idx] = self.heap[idx], self.heap[smallest_idx]
            self.heapify(smallest_idx)

    def peek(self):
        if self.size > 0:
            return self.heap[1]
        return False

    def pop(self):
        if self.size == 0:
            return False

        top = self.heap[1]
        self.heap[1] = self.heap[self.size]
        self.heap[self.size] = None
        self.size -= 1
        self.heapify(1)

        return top.value

    def heap_increase_key(self, idx, key):
        if idx < 1 or idx > self.size:
            print("Índice fuera de rango.")
            return False

        if key > self.heap[idx].key:
            print("La nueva clave debe ser menor que la clave actual.")
            return False

        self.heap[idx].key = key
        while idx > 1 and self.heap[self.parent(idx)].key > self.heap[idx].key:
            self.heap[idx], self.heap[self.parent(idx)] = self.heap[self.parent(idx)], self.heap[idx]
            idx = self.parent(idx)

    def heap_insert(self, key, value):
        self.size += 1
        if self.size >= len(self.heap):
            self.heap.append(HeapNode(math.inf, value))
        else:
            self.heap[self.size] = HeapNode(math.inf, value)

        self.heap_increase_key(self.size, key)

    def get_size(self):
        return self.size