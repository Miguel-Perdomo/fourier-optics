import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
import os
from scipy.signal import convolve2d
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# ================== FOURIER ==================
def FT2(u):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))

def IFT2(U):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U)))

# ================== MÁSCARAS ==================
def square_mask(shape, keep_frac):
    Ny, Nx = shape
    wy = max(1, int(np.floor(keep_frac * Ny)))
    wx = max(1, int(np.floor(keep_frac * Nx)))
    if wy % 2 == 0: wy += 1
    if wx % 2 == 0: wx += 1
    mask = np.zeros((Ny, Nx), dtype=np.float32)
    cy, cx = Ny // 2, Nx // 2
    mask[cy-wy//2:cy+wy//2+1, cx-wx//2:cx+wx//2+1] = 1.0
    return mask

def circular_mask(shape, keep_frac):
    Ny, Nx = shape
    Y, X = np.ogrid[:Ny, :Nx]
    cy, cx = Ny//2, Nx//2
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    rmax = min(Ny, Nx) / 2
    R = keep_frac * rmax
    return (r <= R).astype(np.float32)

def gaussian_mask(shape, keep_frac):
    """Máscara gaussiana para filtrado en frecuencia"""
    Ny, Nx = shape
    cy, cx = Ny//2, Nx//2
    y, x = np.ogrid[-cy:Ny-cy, -cx:Nx-cx]
    r = np.sqrt(x**2 + y**2)
    rmax = min(Ny, Nx) / 2
    R = keep_frac * rmax
    # Ajustamos sigma para que el filtro sea suave
    sigma = R / 2.5
    gaussian = np.exp(-(r**2) / (2 * sigma**2))
    return gaussian.astype(np.float32)

# ================== KERNEL DESDE MÁSCARA ==================
def kernel_from_mask(M, lobes=1):
    Ny, Nx = M.shape
    psf = np.fft.ifft2(np.fft.ifftshift(M)).real
    psf_cent = np.fft.fftshift(psf)
    wy = int(np.sum(M[M.shape[0]//2, :] > 0))
    wx = int(np.sum(M[:, M.shape[1]//2] > 0))
    wy, wx = max(wy, 1), max(wx, 1)
    main_y, main_x = int(np.ceil(Ny / wy)), int(np.ceil(Nx / wx))
    Ky = 2 * lobes * main_y + 1
    Kx = 2 * lobes * main_x + 1
    cy, cx = Ny // 2, Nx // 2
    y0, y1 = cy - Ky//2, cy + Ky//2 + 1
    x0, x1 = cx - Kx//2, cx + Kx//2 + 1
    K_crop = psf_cent[y0:y1, x0:x1].copy()
    K_crop /= K_crop.sum() + 1e-12
    return K_crop

# ================== FUNCIONES PRINCIPALES ==================
def recon_fourier(u_s, s, mask_kind="square"):
    u = u_s.astype(np.float32)/255.0
    Uc = FT2(u)
    keep_frac = 1.0/float(s)
    if mask_kind == "square":
        M = square_mask(Uc.shape, keep_frac)
    elif mask_kind == "circular":
        M = circular_mask(Uc.shape, keep_frac)
    elif mask_kind == "gaussian":
        M = gaussian_mask(Uc.shape, keep_frac)
    Uc_filt = Uc * M
    u_rec = np.abs(IFT2(Uc_filt))
    return u_rec, M, Uc

def recon_convolution(u_s, s, mask_kind="square", lobes=1):
    u = u_s.astype(np.float32)/255.0
    keep_frac = 1.0/float(s)
    if mask_kind == "square":
        M = square_mask(u.shape, keep_frac)
    elif mask_kind == "circular":
        M = circular_mask(u.shape, keep_frac)
    elif mask_kind == "gaussian":
        M = gaussian_mask(u.shape, keep_frac)
    h = kernel_from_mask(M, lobes=lobes)
    u_rec = convolve2d(u, h, mode="same", boundary="symm")
    return u_rec, h

def show_spectrum(u_s, s, mask_kind="square"):
    u = u_s.astype(np.float32)/255.0
    Uc = FT2(u)
    keep_frac = 1.0/float(s)
    if mask_kind == "square":
        M = square_mask(Uc.shape, keep_frac)
    elif mask_kind == "circular":
        M = circular_mask(Uc.shape, keep_frac)
    elif mask_kind == "gaussian":
        M = gaussian_mask(Uc.shape, keep_frac)
    A = np.log1p(np.abs(Uc))
    vmax = np.percentile(A, 99)
    return A, M, vmax

def compare_all(u_s, s, lobes=1):
    # Fourier cuadrada
    u_f_sq, _, _ = recon_fourier(u_s, s, "square")
    # Fourier circular
    u_f_circ, _, _ = recon_fourier(u_s, s, "circular")
    # Fourier gaussiana
    u_f_gauss, _, _ = recon_fourier(u_s, s, "gaussian")
    # Convolución cuadrada
    u_c_sq, _ = recon_convolution(u_s, s, "square", lobes)
    # Convolución circular
    u_c_circ, _ = recon_convolution(u_s, s, "circular", lobes)
    # Convolución gaussiana
    u_c_gauss, _ = recon_convolution(u_s, s, "gaussian", lobes)
    return u_f_sq, u_f_circ, u_f_gauss, u_c_sq, u_c_circ, u_c_gauss

def compare_two(u_s, s, choice1, choice2, lobes=1):
    def run_choice(ch):
        if ch=="F1":
            return recon_fourier(u_s, s, "square")[0], "Fourier Cuadrada"
        elif ch=="F2":
            return recon_fourier(u_s, s, "circular")[0], "Fourier Circular"
        elif ch=="F3":
            return recon_fourier(u_s, s, "gaussian")[0], "Fourier Gaussiana"
        elif ch=="C1":
            return recon_convolution(u_s, s, "square", lobes)[0], "Convolución Cuadrada"
        elif ch=="C2":
            return recon_convolution(u_s, s, "circular", lobes)[0], "Convolución Circular"
        elif ch=="C3":
            return recon_convolution(u_s, s, "gaussian", lobes)[0], "Convolución Gaussiana"

    img1, title1 = run_choice(choice1)
    img2, title2 = run_choice(choice2)
    return img1, title1, img2, title2

# ================== FUNCIONES DE EJEMPLOS ==================
def create_radial_gradient(size=400):
    """Crea un gradiente radial"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    img = (1 - R) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def create_sinusoidal_pattern(size=400, freq=10):
    """Crea un patrón sinusoidal"""
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    img = (np.sin(X * freq) * np.cos(Y * freq) + 1) * 127.5
    return img.astype(np.uint8)

def create_checkerboard(size=400, squares=10):
    """Crea un patrón de tablero de ajedrez"""
    pattern = np.zeros((size, size), dtype=np.uint8)
    square_size = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                pattern[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255
    return pattern

def create_multiscale_pattern(size=400):
    """Crea un patrón con múltiples escalas"""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Círculos concéntricos
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = np.sqrt((x - center)**2 + (y - center)**2)
    circles = (np.sin(radius / 10) + 1) * 127.5
    
    # Líneas diagonales
    lines = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            lines[i, j] = ((i + j) % 20) * 12.75
    
    # Combinar patrones
    img = (circles * 0.5 + lines * 0.5).astype(np.uint8)
    return img

# ================== INTERFAZ GRÁFICA ==================
class FourierOpticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Taller 1 - Fourier Óptica")
        self.root.geometry("1200x800")
        
        # Configurar estilo con colores azules
        self.setup_style()
        
        # Variables de control
        self.s_var = tk.IntVar(value=3)
        self.lobes_var = tk.IntVar(value=1)
        self.mask_type_var = tk.StringVar(value="square")
        self.method1_var = tk.StringVar(value="F1")
        self.method2_var = tk.StringVar(value="F2")
        self.img_gray = None
        self.original_img = None  # Para guardar la imagen original del gato
        self.sampled_img = None
        self.last_comparison = None  # Para almacenar la última comparación
        
        # Cargar imagen por defecto
        self.load_default_image()
        
        # Crear interfaz
        self.create_widgets()
        
    def setup_style(self):
        """Configura el estilo visual con colores azules"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores azules
        style.configure('TFrame', background='#e6f2ff')
        style.configure('TLabelframe', background='#e6f2ff', foreground='#003366')
        style.configure('TLabelframe.Label', background='#e6f2ff', foreground='#003366')
        style.configure('TButton', background='#4a86e8', foreground='white')
        style.map('TButton', background=[('active', '#3a76d8')])
        style.configure('TLabel', background='#e6f2ff', foreground='#003366')
        style.configure('TSpinbox', fieldbackground='white', foreground='#003366')
        style.configure('TCombobox', fieldbackground='white', foreground='#003366')
        
    def load_default_image(self):
        # Rutas alternativas para buscar la imagen
        possible_paths = [
            "/home/miguel-perdomo/fourier-optics/talleres/taller1/results/Imagen_prueba_gray.png",
            os.path.join(os.path.expanduser("~"), "Imagen_prueba_gray.png"),
            os.path.join(os.path.dirname(__file__), "Imagen_prueba_gray.png"),
            os.path.join(os.path.dirname(__file__), "..", "results", "Imagen_prueba_gray.png")
        ]
        
        for img_path in possible_paths:
            if os.path.exists(img_path):
                self.img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                self.original_img = self.img_gray.copy()  # Guardar copia de la imagen original
                if self.img_gray is not None:
                    self.update_sampled_image()
                    return
        
        # Si no se encuentra la imagen original, crear una imagen de ejemplo
        self.create_sample_image()
        self.original_img = self.img_gray.copy()  # Guardar copia de la imagen original
    
    def create_sample_image(self):
        # Crear una imagen de ejemplo con patrones sencillos
        width, height = 400, 400
        self.img_gray = np.zeros((height, width), dtype=np.uint8)
        
        # Añadir algunos patrones
        cv2.rectangle(self.img_gray, (50, 50), (150, 150), 255, -1)
        cv2.circle(self.img_gray, (300, 100), 50, 200, -1)
        cv2.line(self.img_gray, (100, 300), (300, 300), 150, 5)
        
        # Añadir texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.img_gray, 'Ejemplo', (120, 350), font, 1, 180, 2)
        
        self.update_sampled_image()
    
    def update_sampled_image(self):
        if self.img_gray is not None:
            s = self.s_var.get()
            mask = np.zeros_like(self.img_gray, dtype=np.uint8)
            mask[::s, ::s] = 1
            self.sampled_img = self.img_gray * mask
    
    def create_widgets(self):
        # Panel de control
        control_frame = ttk.LabelFrame(self.root, text="Controles", padding=(10, 10))
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Selector de imagen
        ttk.Button(control_frame, text="Cargar imagen", command=self.load_image).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Imagen gato", command=self.restore_original_image).pack(pady=5, fill=tk.X)
        
        # Ejemplos adicionales
        ttk.Label(control_frame, text="Ejemplos:").pack(pady=(10, 0))
        example_frame = ttk.Frame(control_frame)
        example_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(example_frame, text="Gradiente", command=lambda: self.load_example("gradient")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(example_frame, text="Sinusoidal", command=lambda: self.load_example("sinusoidal")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(example_frame, text="Ajedrez", command=lambda: self.load_example("checkerboard")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(example_frame, text="Multiescala", command=lambda: self.load_example("multiscale")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Factor de muestreo
        ttk.Label(control_frame, text="Factor de muestreo (s):").pack(pady=(10, 0))
        s_spin = ttk.Spinbox(control_frame, from_=1, to=20, textvariable=self.s_var, command=self.update_sampled_image)
        s_spin.pack(pady=5, fill=tk.X)
        
        # Número de lóbulos
        ttk.Label(control_frame, text="Número de lóbulos:").pack(pady=(10, 0))
        lobes_spin = ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.lobes_var)
        lobes_spin.pack(pady=5, fill=tk.X)
        
        # Tipo de máscara
        ttk.Label(control_frame, text="Tipo de máscara:").pack(pady=(10, 0))
        mask_combo = ttk.Combobox(control_frame, textvariable=self.mask_type_var, 
                                 values=["square", "circular", "gaussian"], state="readonly")
        mask_combo.pack(pady=5, fill=tk.X)
        
        # Métodos para comparación
        ttk.Label(control_frame, text="Método 1:").pack(pady=(10, 0))
        method1_combo = ttk.Combobox(control_frame, textvariable=self.method1_var, 
                                    values=["F1", "F2", "F3", "C1", "C2", "C3"], state="readonly")
        method1_combo.pack(pady=5, fill=tk.X)
        
        ttk.Label(control_frame, text="Método 2:").pack(pady=(10, 0))
        method2_combo = ttk.Combobox(control_frame, textvariable=self.method2_var, 
                                    values=["F1", "F2", "F3", "C1", "C2", "C3"], state="readonly")
        method2_combo.pack(pady=5, fill=tk.X)
        
        # Botones de acción
        ttk.Button(control_frame, text="Imagen original", command=self.show_original).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Imagen muestreada", command=self.show_sampled).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Espectro + máscara", command=self.show_spectrum).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Reconstrucción", command=self.show_reconstruction).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Comparar dos métodos", command=self.compare_two_methods).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Comparar todos", command=self.compare_all_methods).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Perfil de intensidad", command=self.show_intensity_profile).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Visualizar Kernel", command=self.show_kernel).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Salir", command=self.root.quit).pack(pady=5, fill=tk.X)
        
        # Área de visualización
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Barra de herramientas
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def load_image(self):
        # Lista más completa de formatos de imagen
        file_types = [
            ("Todos los formatos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif;*.ppm;*.pgm;*.pbm;*.webp"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg;*.jpeg"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff;*.tif"),
            ("Todos los archivos", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Intentar leer la imagen en escala de grises
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                # Si no funciona, intentar leer como color y convertir
                if img is None:
                    img_color = cv2.imread(file_path)
                    if img_color is not None:
                        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                
                if img is not None:
                    self.img_gray = img
                    self.update_sampled_image()
                    messagebox.showinfo("Éxito", "Imagen cargada correctamente")
                else:
                    messagebox.showerror("Error", "No se pudo cargar la imagen. Formato no compatible.")
                    
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def restore_original_image(self):
        """Restaura la imagen original del gato"""
        if self.original_img is not None:
            self.img_gray = self.original_img.copy()
            self.update_sampled_image()
            self.show_original()
            messagebox.showinfo("Imagen restaurada", "Se ha restaurado la imagen original del gato")
        else:
            messagebox.showerror("Error", "No hay imagen original guardada")
    
    def load_example(self, example_type):
        """Carga un ejemplo predefinido"""
        if example_type == "gradient":
            self.img_gray = create_radial_gradient()
        elif example_type == "sinusoidal":
            self.img_gray = create_sinusoidal_pattern()
        elif example_type == "checkerboard":
            self.img_gray = create_checkerboard()
        elif example_type == "multiscale":
            self.img_gray = create_multiscale_pattern()
        
        self.update_sampled_image()
        self.show_original()
        messagebox.showinfo("Ejemplo cargado", f"Se ha cargado el ejemplo: {example_type}")
    
    def show_original(self):
        if self.img_gray is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.img_gray, cmap='gray')
        ax.set_title("Imagen Original", color='#003366')
        ax.axis("off")
        self.canvas.draw()
    
    def show_sampled(self):
        if self.img_gray is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        self.update_sampled_image()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.sampled_img, cmap='gray')
        ax.set_title(f"Imagen muestreada (s={self.s_var.get()})", color='#003366')
        ax.axis("off")
        self.canvas.draw()
    
    def show_spectrum(self):
        if self.sampled_img is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        s = self.s_var.get()
        mask_kind = self.mask_type_var.get()
        A, M, vmax = show_spectrum(self.sampled_img, s, mask_kind)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(A, cmap='gray', vmax=vmax)
        ax.contour(M, colors='red', linewidths=1)
        ax.set_title(f"Espectro + máscara {mask_kind}", color='#003366')
        ax.axis("off")
        self.canvas.draw()
    
    def show_reconstruction(self):
        if self.sampled_img is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        s = self.s_var.get()
        mask_kind = self.mask_type_var.get()
        method = self.method1_var.get()
        
        if method.startswith("F"):
            u_rec, _, _ = recon_fourier(self.sampled_img, s, mask_kind)
            title = f"Reconstrucción Fourier - {mask_kind}"
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(u_rec, cmap='gray')
            ax.set_title(title, color='#003366')
            ax.axis("off")
            self.canvas.draw()
            
        else:
            lobes = self.lobes_var.get()
            u_rec, h = recon_convolution(self.sampled_img, s, mask_kind, lobes)
            title = f"Reconstrucción Convolución - {mask_kind} (lóbulos={lobes})"
            
            # Preguntar si se quiere ver la visualización alternativa
            if messagebox.askyesno("Visualización alternativa", 
                                  "¿Desea ver la visualización alternativa con escala común?"):
                self.show_alternative_view(s, mask_kind, lobes, u_rec, h)
            else:
                # Mostrar la reconstrucción por convolución normal
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.imshow(u_rec, cmap='gray')
                ax.set_title(title, color='#003366')
                ax.axis("off")
                self.canvas.draw()
    
    def show_alternative_view(self, s, mask_kind, lobes, u_conv, kernel):
        """Muestra la visualización alternativa con escala común"""
        # Calcular la reconstrucción por Fourier para comparación
        u_fourier, _, _ = recon_fourier(self.sampled_img, s, mask_kind)
        
        # Escala común para Fourier y Convolución
        vmin, vmax = 0, u_fourier.max()
        
        # Visualización: 4 imágenes
        self.figure.clear()
        
        # 1) Imagen muestreada
        ax1 = self.figure.add_subplot(221)
        ax1.imshow(self.sampled_img, cmap='gray')
        ax1.set_title(f'Imagen muestreada (s={s})', color='#003366')
        ax1.axis('off')
        
        # 2) Reconstrucción Fourier
        ax2 = self.figure.add_subplot(222)
        im_fourier = ax2.imshow(u_fourier, cmap='gray', norm=Normalize(vmin=vmin, vmax=vmax))
        ax2.set_title('Reconstrucción Fourier', color='#003366')
        ax2.axis('off')
        self.figure.colorbar(im_fourier, ax=ax2)
        
        # 3) Reconstrucción por convolución
        ax3 = self.figure.add_subplot(223)
        im_conv = ax3.imshow(u_conv, cmap='gray', norm=Normalize(vmin=vmin, vmax=vmax))
        ax3.set_title(f"Reconstrucción Convolución (lobes={lobes})", color='#003366')
        ax3.axis('off')
        self.figure.colorbar(im_conv, ax=ax3)
        
        # 4) Kernel
        ax4 = self.figure.add_subplot(224)
        im_kernel = ax4.imshow(kernel, cmap='viridis')
        ax4.set_title(f"Kernel ({mask_kind})", color='#003366')
        ax4.axis('off')
        self.figure.colorbar(im_kernel, ax=ax4)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def compare_two_methods(self):
        if self.sampled_img is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        s = self.s_var.get()
        lobes = self.lobes_var.get()
        method1 = self.method1_var.get()
        method2 = self.method2_var.get()
        
        img1, title1, img2, title2 = compare_two(self.sampled_img, s, method1, method2, lobes)
        
        # Guardar la comparación para el perfil de intensidad
        self.last_comparison = (img1, title1, img2, title2)
        
        # Preguntar por visualización alternativa si hay convoluciones
        if method1.startswith("C") or method2.startswith("C"):
            if messagebox.askyesno("Visualización alternativa", 
                                  "¿Desea aplicar visualización alternativa a las convoluciones?"):
                # Recalcular con escala común si es necesario
                if method1.startswith("C"):
                    u_fourier, _, _ = recon_fourier(self.sampled_img, s, "circular" if "2" in method1 else "square")
                    vmin, vmax = 0, u_fourier.max()
                    img1 = np.clip(img1, vmin, vmax)
                    img1 = (img1 - vmin) / (vmax - vmin)
                
                if method2.startswith("C"):
                    u_fourier, _, _ = recon_fourier(self.sampled_img, s, "circular" if "2" in method2 else "square")
                    vmin, vmax = 0, u_fourier.max()
                    img2 = np.clip(img2, vmin, vmax)
                    img2 = (img2 - vmin) / (vmax - vmin)
        
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax1.imshow(img1, cmap='gray')
        ax1.set_title(title1, color='#003366')
        ax1.axis("off")
        
        ax2 = self.figure.add_subplot(122)
        ax2.imshow(img2, cmap='gray')
        ax2.set_title(title2, color='#003366')
        ax2.axis("off")
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_intensity_profile(self):
        if self.last_comparison is None:
            messagebox.showerror("Error", "Primero debe comparar dos métodos")
            return
            
        img1, title1, img2, title2 = self.last_comparison
        
        # Mostrar el perfil en la figura principal
        self.figure.clear()
        
        # Imágenes
        ax1 = self.figure.add_subplot(221)
        ax1.imshow(img1, cmap='gray')
        ax1.set_title(title1, color='#003366', fontsize=10)
        ax1.axis("off")
        
        ax2 = self.figure.add_subplot(222)
        ax2.imshow(img2, cmap='gray')
        ax2.set_title(title2, color='#003366', fontsize=10)
        ax2.axis("off")
        
        # Perfil de intensidad
        ax3 = self.figure.add_subplot(212)
        Ny, Nx = img1.shape
        cy = Ny//2
        perfil1 = img1[cy,:]
        perfil2 = img2[cy,:]
        x = np.arange(Nx)
        
        ax3.plot(x, perfil1, label=title1, color='#4a86e8')
        ax3.plot(x, perfil2, label=title2, color='#e69138', linestyle="--")
        ax3.set_title("Perfil de intensidades (horizontal, centro)", color='#003366')
        ax3.set_xlabel("Posición (pixeles)", color='#003366')
        ax3.set_ylabel("Intensidad (u.a.)", color='#003366')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_kernel(self):
        if self.img_gray is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        s = self.s_var.get()
        mask_kind = self.mask_type_var.get()
        lobes = self.lobes_var.get()
        
        # Crear máscara
        keep_frac = 1.0/float(s)
        if mask_kind == "square":
            M = square_mask(self.img_gray.shape, keep_frac)
        elif mask_kind == "circular":
            M = circular_mask(self.img_gray.shape, keep_frac)
        elif mask_kind == "gaussian":
            M = gaussian_mask(self.img_gray.shape, keep_frac)
        
        # Obtener kernel
        h = kernel_from_mask(M, lobes=lobes)
        
        # Mostrar información del kernel
        kernel_info = f"Tamaño del kernel: {h.shape}\nSuma de valores: {h.sum():.4f}"
        
        # Crear ventana para mostrar el kernel
        kernel_window = tk.Toplevel(self.root)
        kernel_window.title("Visualización del Kernel")
        kernel_window.geometry("1000x600")
        
        # Frame principal
        main_frame = ttk.Frame(kernel_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Información del kernel
        info_label = ttk.Label(main_frame, text=kernel_info)
        info_label.pack(pady=10)
        
        # Figura para visualizar el kernel
        kernel_fig = Figure(figsize=(10, 6))
        kernel_canvas = FigureCanvasTkAgg(kernel_fig, main_frame)
        kernel_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Visualizar el kernel
        ax1 = kernel_fig.add_subplot(121)
        im = ax1.imshow(h, cmap='viridis')
        ax1.set_title("Kernel 2D", color='#003366')
        kernel_fig.colorbar(im, ax=ax1)
        
        # Perfil del kernel
        ax2 = kernel_fig.add_subplot(122)
        center_y, center_x = h.shape[0] // 2, h.shape[1] // 2
        profile = h[center_y, :] if h.shape[0] > 1 else h[0, :]
        x = np.arange(len(profile))
        ax2.plot(x, profile, color='#4a86e8')
        ax2.set_title("Perfil del Kernel (línea central)", color='#003366')
        ax2.set_xlabel("Posición", color='#003366')
        ax2.set_ylabel("Valor", color='#003366')
        ax2.grid(alpha=0.3)
        
        kernel_fig.tight_layout()
        kernel_canvas.draw()
    
    def compare_all_methods(self):
        if self.sampled_img is None:
            messagebox.showerror("Error", "Primero debe cargar una imagen")
            return
            
        s = self.s_var.get()
        lobes = self.lobes_var.get()
        
        u_f_sq, u_f_circ, u_f_gauss, u_c_sq, u_c_circ, u_c_gauss = compare_all(self.sampled_img, s, lobes)
        
        # Preguntar por visualización alternativa para convoluciones
        if messagebox.askyesno("Visualización alternativa", 
                              "¿Desea aplicar visualización alternativa a las convoluciones?"):
            # Recalcular con escala común
            u_fourier, _, _ = recon_fourier(self.sampled_img, s, "circular")
            vmin, vmax = 0, u_fourier.max()
            u_c_sq = np.clip(u_c_sq, vmin, vmax)
            u_c_sq = (u_c_sq - vmin) / (vmax - vmin)
            u_c_circ = np.clip(u_c_circ, vmin, vmax)
            u_c_circ =  (u_c_circ - vmin) / (vmax - vmin)
            u_c_gauss = np.clip(u_c_gauss, vmin, vmax)
            u_c_gauss = (u_c_gauss - vmin) / (vmax - vmin)
        
        self.figure.clear()
        ax1 = self.figure.add_subplot(221)
        ax1.imshow(u_f_sq, cmap='gray')
        ax1.set_title("Fourier - Cuadrada", color='#003366', fontsize=8)
        ax1.axis("off")
        
        ax2 = self.figure.add_subplot(222)
        ax2.imshow(u_f_circ, cmap='gray')
        ax2.set_title("Fourier - Circular", color='#003366', fontsize=8)
        ax2.axis("off")
        
        ax3 = self.figure.add_subplot(223)
        ax3.imshow(u_c_sq, cmap='gray')
        ax3.set_title("Convolución - Cuadrada", color='#003366', fontsize=8)
        ax3.axis("off")
        
        ax4 = self.figure.add_subplot(224)
        ax4.imshow(u_c_circ, cmap='gray')
        ax4.set_title("Convolución - Circular", color='#003366', fontsize=8)
        ax4.axis("off")
        
        
        self.figure.tight_layout()
        self.canvas.draw()

# ================== MAIN ==================
if __name__ == "__main__":
    root = tk.Tk()
    app = FourierOpticsApp(root)
    root.mainloop()