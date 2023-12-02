import tkinter as tk
from tkinter import filedialog,messagebox
import heapq
from collections import Counter
import os
from PIL import Image
import numpy as np
import pickle
import wave

class FileCompressionApp:
    
    #Constructor
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

    #Main loop
    def run(self):
        self.root.mainloop()
        
    #Encoding and decoding methods
    def huffman_encoding(self, data):
        if not data:
            return None, None
        
        # Calculate the frequency of each symbol in the input data
        frequency = Counter(data)
        
        # Create a list of lists, where each inner list contains a symbol and its Huffman code
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        
        # Create a binary heap (min-heap) from the list
        heapq.heapify(heap)
        
        # Build the Huffman tree by repeatedly merging the two lowest-weight nodes
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        # Create a dictionary that maps symbols to their Huffman codes
        huffman_dict = {item[0]: item[1] for item in heap[0][1:]}
        
        # Encode the input data using the Huffman codes
        encoded_data = ''.join(huffman_dict[symbol] for symbol in data)
        
        return huffman_dict, encoded_data

    def huffman_decoding(self, encoded_data, huffman_tree):
        if not encoded_data or not huffman_tree:
            return ""
        
        # Crear un diccionario invertido que mapea códigos de Huffman a símbolos
        reversed_tree = {v: k for k, v in huffman_tree.items()}
        
        # Inicializar variables para la decodificación
        decoded_data = ""
        current_code = ""
        
        # Iterar a través de los datos codificados, construyendo la salida decodificada
        for bit in encoded_data:
            current_code += bit
            if current_code in reversed_tree:
                # Asegúrate de convertir el valor en una cadena antes de concatenar
                decoded_data += str(reversed_tree[current_code])
                current_code = ""
        
        return decoded_data

    def huffman_decoding_images(self, encoded_data, huffman_tree):
        if not encoded_data or not huffman_tree:
            return []

        # Diccionario invertido: códigos a tuplas
        reversed_tree = {v: k for k, v in huffman_tree.items()}

        # Decodificación
        decoded_data = []
        current_code = ""
        for bit in encoded_data:
            current_code += bit
            if current_code in reversed_tree:
                decoded_data.append(reversed_tree[current_code])
                current_code = ""

        return decoded_data
    
    #Encoding and decoding for audio 
    def huffman_encoding_audio(self, audio_array):
        if audio_array.size == 0:
            return None, None

        # Convertir el array de NumPy a una lista de símbolos (muestras de audio)
        # audio_bytes = audio_array.astype(np.int8).tobytes()
        
        audio_bytes = audio_array.tobytes()

        # Calcular la frecuencia de cada símbolo en los datos de audio
        frequency = Counter(audio_bytes)

        # Crear una lista de listas, donde cada inner list contiene un símbolo y su código Huffman
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]

        # Crear un heap binario (min-heap) a partir de la lista
        heapq.heapify(heap)

        # Construir el árbol de Huffman fusionando repetidamente los dos nodos de menor peso
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        # Crear un diccionario que mapea símbolos a sus códigos Huffman
        huffman_dict = {item[0]: item[1] for item in heap[0][1:]}

        # Codificar los datos de audio utilizando los códigos Huffman
        encoded_data = ''.join(huffman_dict[symbol] for symbol in audio_bytes)

        return huffman_dict, encoded_data

    def huffman_decoding_audio(self, encoded_data, huffman_dict):
            if not encoded_data or not huffman_dict:
                return np.array([])  # Devolver un array vacío en caso de datos vacíos o diccionario vacío

            # Invertir el diccionario de Huffman para mapear códigos a símbolos
            reversed_dict = {v: k for k, v in huffman_dict.items()}

            # Inicializar variables para la decodificación
            decoded_data = []
            current_code = ""

            # Iterar a través de los datos codificados, construyendo la salida decodificada
            for bit in encoded_data:
                current_code += bit
                if current_code in reversed_dict:
                    decoded_data.append(reversed_dict[current_code])
                    current_code = ""

            # Convertir la lista de símbolos decodificados de nuevo a un array de NumPy
            decoded_array = np.frombuffer(bytearray(decoded_data), dtype=np.int16)

            return decoded_array

    #Encoding and decoding for video

    def huffman_encoding_video(self, data):
        if not data:
            return None, None

        # Trata los datos como una secuencia de bytes
        if isinstance(data, str):
            data = data.encode()

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
        encoded_data = ''.join(huffman_dict[byte] for byte in data)

        return huffman_dict, encoded_data

    def huffman_decoding_video(self, encoded_data, huffman_tree):
        if not encoded_data or not huffman_tree:
            return b""

        reversed_tree = {v: k for k, v in huffman_tree.items()}

        decoded_data = []
        current_code = ""
        for bit in encoded_data:
            current_code += bit
            if current_code in reversed_tree:
                decoded_data.append(reversed_tree[current_code])
                current_code = ""

        # Convertir la lista de bytes decodificados de nuevo a un objeto de bytes
        return bytearray(decoded_data)


    #Ui methods
    def upload_file(self):
        # Open a file dialog to select a file and update the file entry field with the selected file path
        file_path = filedialog.askopenfilename()
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, file_path)

    def process_file(self, file_type):
        # Get the selected file path from the entry field
        file_path = self.file_entry.get()
        if not file_path:
            # Display a warning message if no file is selected
            messagebox.showwarning("Warning", "Ningun archivo seleccionado")
            return
        # Extract the file extension from the file path
        file_extension = os.path.splitext(file_path)[1]
        # Check the file extension to determine whether it's binary (.bin) or RLE (.rle)
        if file_extension == '.bin' or file_extension == '.rle':
            # Depending on the file type and extension, call the appropriate decompression method
            if file_type == 'Audio':
                self.decompress_audio(file_path)
            elif file_type == 'Image':
                self.decompress_image(file_path)
            elif file_type == 'Video':
                self.decompress_video(file_path)
            elif file_type == 'Text':
                self.decompress_text(file_path)
        else:
            # Depending on the file type and extension, call the appropriate compression method
            if file_type == 'Audio':
                self.compress_audio(file_path)
            elif file_type == 'Image':
                self.compress_image(file_path)
            elif file_type == 'Video':
                self.compress_video(file_path)
            elif file_type == 'Text':
                self.compress_text(file_path)

    #Text methods
    def compress_text(self, file_path):
        try:
            # Open and read the input file
            with open(file_path, 'r') as f:
                data = f.read()

            # Encode the data using Huffman coding
            huffman_tree, encoded_data = self.huffman_encoding(data)

            # Check if encoding was successful
            if huffman_tree is None or encoded_data is None:
                messagebox.showerror("Error", "No se pudo comprimir el archivo (Podria estar vacio)")
                return

            # Convert the string of '0's and '1's to bytes
            padded_encoded_data = self.pad_encoded_data(encoded_data)
            b = bytearray()
            for i in range(0, len(padded_encoded_data), 8):
                byte = padded_encoded_data[i:i+8]
                b.append(int(byte, 2))

            # Ask the user to select the location and name of the compressed file
            save_path = filedialog.asksaveasfilename(defaultextension=".bin",
                                                    filetypes=[("Binary files", "*.bin")])

            # Check if the user canceled the operation
            if not save_path:
                messagebox.showwarning("Cancelled", "Se cancelo el proceso")
                return

            # Save the Huffman tree and the compressed data to the binary file
            with open(save_path, 'wb') as f:
                pickle.dump((huffman_tree, bytes(b)), f)
            messagebox.showinfo("Succes", f"Archivo comprimido guardado en: {save_path}")

        except IOError as e:
            messagebox.showerror("Error", f"Error al abrir o leer el archivo {e}")

    def decompress_text(self, file_path):
        try:
            # Open and read the compressed file
            with open(file_path, 'rb') as f:
                huffman_tree, encoded_bytes = pickle.load(f)

            # Convert bytes to the string of '0's and '1's
            encoded_data = ''.join(f"{byte:08b}" for byte in encoded_bytes)

            # Remove padding
            padded_info = encoded_data[:8]
            extra_padding = int(padded_info, 2)
            encoded_data = encoded_data[8:-extra_padding]

            # Decode the Huffman-encoded data
            decoded_data = self.huffman_decoding(encoded_data, huffman_tree)

            # Ask the user to select the location and name of the output file
            save_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                    filetypes=[("Text files", "*.txt")])

            # Check if the user canceled the operation
            if not save_path:
                messagebox.showwarning("Cancelled", "Se cancelo el proceso")
                return

            # Save the decompressed data to the text file
            with open(save_path, "w") as f:
                f.write(decoded_data)
            messagebox.showinfo("Succes", f"Archivo descomprimido guardado en: {save_path}")

        except IOError as e:
            messagebox.showerror("Error", f"Error al abrir o leer el archivo {e}")

    def pad_encoded_data(self, encoded_data):
        # Calculate the extra padding needed to make the encoded data a multiple of 8 bits
        extra_padding = 8 - len(encoded_data) % 8
        # Append '0's to the encoded data to achieve the padding
        for i in range(extra_padding):
            encoded_data += "0"
        # Create an 8-bit binary string representing the amount of padding added
        padded_info = "{0:08b}".format(extra_padding)
        # Combine the padded information and the encoded data
        padded_encoded_data = padded_info + encoded_data
        return padded_encoded_data
    
    #Image methods
    def compress_image(self, file_path):
        # Read the image and convert it to a matrix
        image_matrix = self.image_to_matrix(file_path)

        if image_matrix is None:
            messagebox.showerror("Error", "No se pudo leer la imagen")
            return

        # Store the dimensions of the image (width, height, and channels)
        self.width, self.height, self.channels = image_matrix.shape
        
        # Convertir la imagen en una lista lineal de valores (flattening)
        pixel_list = [tuple(pixel) for row in image_matrix for pixel in row]
                
        # Codificar la imagen usando Huffman
        huffman_dict, encoded_image = self.huffman_encoding(pixel_list)
                
        # Convert the string of '0's and '1's to bytes
        padded_encoded_data = self.pad_encoded_data(encoded_image)
        b = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte = padded_encoded_data[i:i+8]
            b.append(int(byte, 2))
            
        # Ask the user to select the location and name of the compressed image file
        save_path = filedialog.asksaveasfilename(defaultextension=".bin",
                                                filetypes=[("BIN files", "*.bin")])

        if save_path:
            # Save the dimensions and the encoded image to the selected file
            with open(save_path, 'wb') as f:
                pickle.dump((self.width, self.height,self.channels, huffman_dict, bytes(b)), f)
                messagebox.showinfo("Success", f"Imagen comprimida guardada en: {save_path}")

    def decompress_image(self, file_path):
        # Open the compressed image file
        with open(file_path, 'rb') as f:
            width, height,channels,huffman_dict, encoded_image = pickle.load(f)
            
        # Convert bytes to the string of '0's and '1's
        encoded_data = ''.join(f"{byte:08b}" for byte in encoded_image)

        # Remove padding
        padded_info = encoded_data[:8]
        extra_padding = int(padded_info, 2)
        encoded_data = encoded_data[8:-extra_padding]
        
        
        # Decodificar la imagen usando Huffman
        decoded_pixel_list = self.huffman_decoding_images(encoded_data, huffman_dict)
        
        
        decoded_image = [decoded_pixel_list[i:i+width] for i in range(0, len(decoded_pixel_list),width)]
        # Reconvertir la lista en la matriz de la imagen
        image_matrix = np.array(decoded_image).reshape(width,height,channels)

        # Ask the user to select the location and name of the decompressed image file
        save_path = filedialog.asksaveasfilename(defaultextension=".bmp",
                                                filetypes=[("BMP files", "*.bmp")])

        if save_path:
            # Convert the decoded matrix back to an image and save it to the selected file
            self.matrix_to_image(image_matrix, save_path)
            messagebox.showinfo("Success", f"Imagen descomprimida guardada en: {save_path}")

    def image_to_matrix(self, file_path):
        try:
            with Image.open(file_path) as img:
                img_matrix = np.array(img)
                return img_matrix
        except Exception as e:
            # Show an error message if there's an issue with image processing
            messagebox.showerror("Error", f"Error al procesar la imagen {e}")
            return None

    def matrix_to_image(self, matrix, save_path):
        # Convert the matrix to an image and save it to the specified path
        img = Image.fromarray(np.array(matrix, dtype=np.uint8))
        img.save(save_path)

    #Audio methods
    def wav_to_matrix(self, archivo_wav):
        with wave.open(archivo_wav, 'r') as wav_file:
            # Leer parámetros del archivo WAV
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()

            # Leer datos de audio
            audio_data = wav_file.readframes(n_frames)

            # Determinar el tipo de datos adecuado para el array de NumPy
            if sample_width == 1:  # 8-bit WAV
                dtype = np.uint8
            elif sample_width == 2:  # 16-bit WAV
                dtype = np.int16
            else:
                raise ValueError("Profundidad de bits no soportada")

            # Convertir a array de NumPy
            audio_array = np.frombuffer(audio_data, dtype=dtype)

            # Reshape el array si es estéreo
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2)

            return audio_array, (n_channels, frame_rate, sample_width)

    def matrix_to_wav(self, audio_array, archivo_wav, wav_params):
        n_channels, frame_rate, sample_width = wav_params

        # Convertir array de NumPy a bytes
        audio_bytes = audio_array.tobytes()

        # Abrir archivo WAV y escribir muestras con los parámetros originales
        with wave.open(archivo_wav, 'wb') as wav_file:
            wav_file.setparams((n_channels, sample_width, frame_rate, 0, 'NONE', 'NONE'))
            wav_file.writeframes(audio_bytes)

    def compress_audio(self, file_path):
        # Obtener los datos de audio como un array de NumPy y los parámetros del WAV
        audio_array, wav_params = self.wav_to_matrix(file_path)

        # Aplicar la codificación Huffman al array de NumPy
        huffman_tree, encoded_data = self.huffman_encoding_audio(audio_array)
        
        # Convertir la cadena de '0's y '1's en bytes utilizando la función de relleno
        padded_encoded_data = self.pad_encoded_data(encoded_data)
        b = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte = padded_encoded_data[i:i+8]
            b.append(int(byte, 2))

        # Guardar los datos comprimidos, el árbol de Huffman y los parámetros del WAV en un archivo binario
        save_path = filedialog.asksaveasfilename(defaultextension=".bin",
                                                filetypes=[("Binary files", "*.bin")])
        if not save_path:
            messagebox.showwarning("Cancelled", "Se canceló el proceso")
            return

        with open(save_path, 'wb') as f:
            pickle.dump((huffman_tree, bytes(b), wav_params), f)
        messagebox.showinfo("Success", f"Archivo comprimido guardado en: {save_path}")

    def decompress_audio(self, file_path):
        # Leer el archivo binario para obtener el árbol de Huffman, los datos comprimidos y los parámetros del WAV
        try:
            with open(file_path, 'rb') as f:
                huffman_tree, encoded_bytes, wav_params = pickle.load(f)
        except IOError as e:
            messagebox.showerror("Error", f"Error al abrir o leer el archivo {e}")
            return

        # Convertir bytes nuevamente en la cadena de '0's y '1's
        encoded_data = ''.join(f"{byte:08b}" for byte in encoded_bytes)

        # Decodificar los datos comprimidos utilizando la función específica para audio
        decoded_array = self.huffman_decoding_audio(encoded_data, huffman_tree)
        
        # Guardar el array como archivo WAV
        save_path = filedialog.asksaveasfilename(defaultextension=".wav",
                                                filetypes=[("WAV files", "*.wav")])
        if not save_path:
            messagebox.showwarning("Cancelled", "Se canceló el proceso")
            return

        self.matrix_to_wav(decoded_array, save_path, wav_params)
        messagebox.showinfo("Success", f"Archivo descomprimido guardado en: {save_path}")

    #Video methods
    def compress_video(self, file_path):
        video_data = self.read_video(file_path)
        huffman_tree, encoded_video = self.huffman_encoding_video(video_data)

        padded_encoded_data = self.pad_encoded_data(encoded_video)
        byte_array = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte = padded_encoded_data[i:i+8]
            byte_array.append(int(byte, 2))

        save_path = filedialog.asksaveasfilename(defaultextension=".bin",
                                                 filetypes=[("Binary files", "*.bin")])
        if not save_path:
            messagebox.showwarning("Cancelled", "Se canceló el proceso de guardado")
            return

        with open(save_path, 'wb') as f:
            pickle.dump((huffman_tree, bytes(byte_array)), f)

        messagebox.showinfo("Success", f"Video comprimido guardado en: {save_path}")

    def decompress_video(self, file_path):
        with open(file_path, 'rb') as f:
            huffman_tree, encoded_bytes = pickle.load(f)

        encoded_data = ''.join(f"{byte:08b}" for byte in encoded_bytes)
        padded_info = encoded_data[:8]
        extra_padding = int(padded_info, 2)
        encoded_data = encoded_data[8:-extra_padding]

        decoded_video = self.huffman_decoding_video(encoded_data, huffman_tree)

        save_path = filedialog.asksaveasfilename(defaultextension=".avi",
                                                 filetypes=[("AVI files", "*.avi")])
        if not save_path:
            messagebox.showwarning("Cancelled", "Se canceló el proceso de guardado")
            return

        self.write_video(save_path, decoded_video)
        messagebox.showinfo("Success", f"Video descomprimido guardado en: {save_path}")

    def read_video(self, file_path):
        with open(file_path, 'rb') as f:
            video_data = f.read()      
        return video_data

    def write_video(self, file_path, video_data):
        with open(file_path, 'wb') as f:
            f.write(video_data)
        
if __name__ == "__main__":
    app = FileCompressionApp()
    app.run()