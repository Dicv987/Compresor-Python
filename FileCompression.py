import tkinter as tk
from tkinter import filedialog,messagebox
import heapq
from collections import Counter
import os
from PIL import Image
import numpy as np
import pickle

class FileCompressionApp:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("File Compression Tool")
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)
        self.file_entry = tk.Entry(self.frame, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.upload_button = tk.Button(self.frame, text="Upload File", command=self.upload_file)
        self.upload_button.pack(side=tk.LEFT)
        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(padx=10, pady=(0, 10))
        button_types = ["Audio", "Image", "Video", "Text"]
        for t in button_types:
            button = tk.Button(self.buttons_frame, text=t, command=lambda t=t: self.process_file(t))
            button.pack(side=tk.LEFT, padx=5)

    def run(self):
        self.root.mainloop()

    def huffman_encoding(self, data):
        if not data:
            return None, None
        frequency = Counter(data)
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huffman_dict = {item[0]: item[1] for item in heap[0][1:]}
        encoded_data = ''.join(huffman_dict[symbol] for symbol in data)
        return huffman_dict, encoded_data

    def huffman_decoding(self, encoded_data, huffman_tree):
        if not encoded_data or not huffman_tree:
            return ""
        reversed_tree = {v: k for k, v in huffman_tree.items()}
        decoded_data = ""
        current_code = ""
        for bit in encoded_data:
            current_code += bit
            if current_code in reversed_tree:
                decoded_data += reversed_tree[current_code]
                current_code = ""
        return decoded_data

    def rle_encode(self, image):    #Codifica los datos de la imagen en RLE
        encoded = []
        prev_pixel = None
        count = 0
        for row in image:   #Recorre la imagen por filas y cuenta los píxeles
            for pixel in row:
                if np.array_equal(pixel, prev_pixel):
                    count += 1
                else:
                    if prev_pixel is not None:
                        encoded.append((prev_pixel, count))
                    prev_pixel = pixel
                    count = 1
        encoded.append((prev_pixel, count))  # Añadir el último píxel
        return encoded

    def rle_decode(self, encoded_data, dimensions): #Decodifica los datos codificados en RLE
        width, height = dimensions
        # Asegura que haya datos codificados antes de acceder a ellos
        if not encoded_data:
            return np.zeros((height, width, 3), dtype=np.uint8)  # Devuelve una imagen en negro si no hay datos

        # Determinar si es RGB o RGBA basado en la longitud del primer píxel
        channels = len(encoded_data[0][0])
        decoded_image = np.zeros((height, width, channels), dtype=np.uint8)

        x, y = 0, 0
        for (value, count) in encoded_data:
            for _ in range(count):
                decoded_image[y][x] = value
                x += 1
                if x == width:
                    x = 0
                    y += 1
        return decoded_image
    
    def image_to_matrix(self, file_path):   #Convierte la imagen en una matriz
        try:
            with Image.open(file_path) as img:
                img_matrix = np.array(img)
                return img_matrix
        except Exception as e:
            messagebox.showerror("Error",f"Error al procesar la imagen{e}")
            return None
        
    def matrix_to_image(self,matrix, save_path):    #Convierte la matriz en imagen
        img = Image.fromarray(np.array(matrix, dtype=np.uint8))
        img.save(save_path)

    def upload_file(self):
        file_path = filedialog.askopenfilename()
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, file_path)

    def pad_encoded_data(self, encoded_data):
        extra_padding = 8 - len(encoded_data) % 8
        for i in range(extra_padding):
            encoded_data += "0"
        padded_info = "{0:08b}".format(extra_padding)
        padded_encoded_data = padded_info + encoded_data
        return padded_encoded_data

    def process_file(self, file_type):
        file_path = self.file_entry.get()
        if not file_path:
            messagebox.showwarning("Warning","Ningun archivo seleccionado")
            return
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == '.bin' or file_extension == '.rle':
            if file_type == 'Audio':
                self.decompress_audio(file_path)
            elif file_type == 'Image':
                self.decompress_image(file_path)
            elif file_type == 'Video':
                self.decompress_video(file_path)
            elif file_type == 'Text':
                self.decompress_text(file_path)
        else:
            if file_type == 'Audio':
                self.compress_audio(file_path)
            elif file_type == 'Image':
                self.compress_image(file_path)
            elif file_type == 'Video':
                self.compress_video(file_path)
            elif file_type == 'Text':
                self.compress_text(file_path)
    
    def compress_audio(self,file_path):
        print(f"Compressing audio file: {file_path}")
        # Aquí va la lógica para comprimir archivos de audio

    def decompress_audio(self,file_path):
        print(f"Decompressing audio file: {file_path}")
        # Aquí va la lógica para descomprimir archivos de audio
        
    def compress_image(self, file_path):

        image_matrix = self.image_to_matrix(file_path)

        if image_matrix is None:
            messagebox.showerror("Error","No se pudo leer la imagen")
            return
        
        self.width, self.height, self.channels = image_matrix.shape  # Almacenar las dimensiones de la imagen
        encoded_image = self.rle_encode(image_matrix.tolist())
        save_path = filedialog.asksaveasfilename(defaultextension=".rle",
                                                filetypes=[("RLE files", "*.rle")])
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump((self.width, self.height, encoded_image), f)
                messagebox.showinfo("Success", f"Imagen comprimida guardada en: {save_path}")

    def decompress_image(self, file_path):
        with open(file_path, 'rb') as f:
            width, height, encoded_image = pickle.load(f)
        
        # Use the actual dimensions stored in the file
        image_matrix = self.rle_decode(encoded_image, (height, width))
        save_path = filedialog.asksaveasfilename(defaultextension=".bmp",
                                                filetypes=[("BMP files", "*.bmp")])
        if save_path:
            self.matrix_to_image(image_matrix, save_path)
            messagebox.showinfo("Sucess",f"Imagen descomprimida guardada en: {save_path}")

    def compress_video(self,file_path):
        print(f"Compressing video file: {file_path}")
        # Aquí va la lógica para comprimir archivos de video

    def decompress_video(self,file_path):
        print(f"Decompressing video file: {file_path}")
        # Aquí va la lógica para descomprimir archivos de video

    def compress_text(self,file_path):
        try:
            with open(file_path, 'r') as f:
                data = f.read()

            huffman_tree, encoded_data = self.huffman_encoding(data)
            if huffman_tree is None or encoded_data is None:
                messagebox.showerror("Error","No se pudo comprimir el archivo (Podria estar vacio)")
                return

            # Convertir la cadena de '0's y '1's a bytes
            padded_encoded_data = self.pad_encoded_data(encoded_data)
            b = bytearray()
            for i in range(0, len(padded_encoded_data), 8):
                byte = padded_encoded_data[i:i+8]
                b.append(int(byte, 2))

            # Pedir al usuario que seleccione la ubicación y nombre del archivo comprimido
            save_path = filedialog.asksaveasfilename(defaultextension=".bin",
                                                    filetypes=[("Binary files", "*.bin")])
            if not save_path:  # Si el usuario cancela, terminar la función
                messagebox.showwarning("Cancelled","Se cancelo el proceso")
                return

            # Guardar el árbol de Huffman y los datos comprimidos
            with open(save_path, 'wb') as f:
                pickle.dump((huffman_tree, bytes(b)), f)
            messagebox.showinfo("Succes",f"Archivo comprimido guardado en: {save_path}")

        except IOError as e:
            messagebox.showerror("Error",f"Error al abrir o leer el archivo {e}")
    
    def decompress_text(self,file_path):
        try:
            with open(file_path, 'rb') as f:
                huffman_tree, encoded_bytes = pickle.load(f)

            # Convertir bytes a la cadena de '0's y '1's
            encoded_data = ''.join(f"{byte:08b}" for byte in encoded_bytes)

            # Eliminar el relleno
            padded_info = encoded_data[:8]
            extra_padding = int(padded_info, 2)
            encoded_data = encoded_data[8:-extra_padding]

            decoded_data = self.huffman_decoding(encoded_data, huffman_tree)

            # Pedir al usuario que seleccione la ubicación y nombre del archivo de salida
            save_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                    filetypes=[("Text files", "*.txt")])
            if not save_path:  # Si el usuario cancela, terminar la función
                messagebox.showwarning("Cancelled","Se cancelo el proceso")
                return

            # Guardar los datos descomprimidos en el archivo seleccionado
            with open(save_path, "w") as f:
                f.write(decoded_data)
            messagebox.showinfo("Succes",f"Archivo descomprimido guardado en: {save_path}")

        except IOError as e:
            messagebox.showerror("Error",f"Error al abrir o leer el archivo {e}")

if __name__ == "__main__":
    app = FileCompressionApp()
    app.run()
