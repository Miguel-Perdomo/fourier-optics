"""
Taller 1 - Fourier Óptica
Autor: Miguel Andrés Perdomo-Gutiérrez - Estefania Velasquez
Descripción:
    Script interactivo con menú:
    1. Mostrar imagen original
    2. Mostrar imagen muestreada
    3. Reconstrucción (Fourier o Convolución)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2, os, unicodedata

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

# ================== KERNELS ==================

def sinc_kernel_square_full(Ny, Nx, s):
    k = 1.0 / float(s)
    m = np.arange(Ny) - Ny//2
    n = np.arange(Nx) - Nx//2
    hm = np.sinc(k*m)
    hn = np.sinc(k*n)
    h  = (k**2) * np.outer(hm, hn)
    h /= h.sum()
    return h.astype(np.float32)

def j1_stable(x, x_switch=8.0, terms=40):
    """
    J1(x) estable:
    - |x| < x_switch: serie por recurrencia (sin factoriales)
    - |x| >= x_switch: asintótica de Debye (2 términos)
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    # Región pequeña: serie por recurrencia
    small = np.abs(x) < x_switch
    xs = x[small]
    if xs.size:
        t = xs * 0.5
        # J1(x) = sum_{m=0}^\infty (-1)^m / (m!(m+1)!) * (x/2)^{2m+1}
        # Recurrencia de términos: term_{m+1} = term_m * (-(t^2) / ((m+1)(m+2)))
        res = np.zeros_like(xs)
        term = t.copy()  # m=0
        res += term
        for m in range(1, terms):
            term *= -(t*t) / (m*(m+1))
            res += term
        out[small] = res

    # Región grande: asintótica (Debye) con 2 términos
    large = ~small
    xl = x[large]
    if xl.size:
        phi = xl - 3.0*np.pi/4.0
        # J1(x) ≈ sqrt(2/(πx)) [cos(phi) - 3/(8x) sin(phi)]
        out[large] = np.sqrt(2.0/(np.pi*xl)) * (np.cos(phi) - (3.0/(8.0*xl))*np.sin(phi))

    return out

def jinc_kernel_np(Ny, Nx, s):
    """
    Kernel jinc 2D para ventana circular (disco) en frecuencia.
    h(r) = 2R * J1(2πR r) / r, con R = 1/(2s) (ciclos/píxel), h(0)=2πR^2.
    """
    R = 1.0/(2.0*float(s))  # radio de corte en ciclos/píxel

    Y, X = np.mgrid[:Ny, :Nx]
    cy, cx = Ny//2, Nx//2
    r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)

    z = 2.0*np.pi*R*r
    h = np.zeros_like(r, dtype=np.float64)

    mask = r > 0
    # jinc con J1 estable
    h[mask] = 2.0*R * j1_stable(z[mask]) / r[mask]
    # valor límite en el centro (r->0)
    h[~mask] = 2.0*np.pi*(R**2)

    # sanear numéricos y normalizar (ganancia DC = 1)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    h /= (h.sum() + 1e-12)

    return h.astype(np.float32)

# ================== CONVOLUCIÓN ==================

def convolve_via_fft(u, h):
    Ny, Nx = u.shape
    Hy, Hx = h.shape
    if (Hy, Hx) != (Ny, Nx):
        Hpad = np.zeros((Ny, Nx), dtype=np.float32)
        cy, cx = Ny//2, Nx//2
        hy, hx = Hy//2, Hx//2
        Hpad[cy-hy:cy+hy+1, cx-hx:cx+hx+1] = h
        h = Hpad
    U  = FT2(u)
    H  = FT2(h)
    Y  = U * H
    return np.clip(IFT2(Y).real, 0.0, 1.0)

# ================== FUNCIONES PRINCIPALES ==================

def show_original(img_gray):
    plt.imshow(img_gray, cmap="gray")
    plt.title("Imagen Original")
    plt.axis("off")
    plt.show()

def show_sampled(img_gray, s):
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[::s, ::s] = 1
    sampled = img_gray * mask
    plt.imshow(sampled, cmap="gray")
    plt.title(f"Imagen muestreada (s={s})")
    plt.axis("off")
    plt.show()
    return sampled

def recon_fourier(u_s, s, mask_kind_text="square"):
    mask_kind = pick_choice(
        mask_kind_text,
        mapping={
            "square":   ["square","cuadrada","cuadrado","rect","rectangular","q","s"],
            "circular": ["circular","circulo","círculo","circle","disk","d"]
        },
        default_key="square"
    )
    u = u_s.astype(np.float32)/255.0
    Uc = FT2(u)
    keep_frac = 1.0/float(s)
    if mask_kind == "square":
        M = square_mask(Uc.shape, keep_frac)
    else:
        M = circular_mask(Uc.shape, keep_frac)
    print(f"[Fourier] s={s} | máscara aplicada: {mask_kind}")
    Uc_filt = Uc * M
    u_rec = np.clip(IFT2(Uc_filt).real, 0.0, 1.0)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(u, cmap="gray"); plt.title(f"Muestreada (s={s})"); plt.axis("off")
    A = np.abs(Uc); vmax=np.percentile(A,99)
    plt.subplot(1,3,2); plt.imshow(A, cmap="gray", vmax=vmax); plt.contour(M, colors="red", linewidths=1)
    plt.title(f"|FFT| + máscara: {mask_kind}"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(u_rec, cmap="gray"); plt.title("Reconstrucción Fourier"); plt.axis("off")
    plt.show()

def recon_convolution(u_s, s, kernel_text="sinc"):
    kernel_kind = pick_choice(
        kernel_text,
        mapping={
            "sinc": ["sinc","cuadrada","cuadrado","square","rect","q","s"],
            "jinc": ["jinc","circular","circulo","círculo","circle","airy","bessel","d"]
        },
        default_key="sinc"
    )
    u = u_s.astype(np.float32)/255.0
    if kernel_kind == "sinc":
        h = sinc_kernel_square_full(*u.shape, s)
    else:
        h = jinc_kernel_np(*u.shape, s)
    print(f"[Convolución] s={s} | kernel aplicado: {kernel_kind}")
    u_rec = convolve_via_fft(u, h)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(u, cmap="gray"); plt.title(f"Muestreada (s={s})"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(h/(h.max()+1e-12), cmap="gray"); plt.title(f"Kernel {kernel_kind}"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(u_rec, cmap="gray"); plt.title("Reconstrucción Convolución"); plt.axis("off")
    plt.show()

# ================== Helpers ==================

def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s

def pick_choice(user_text: str, mapping: dict, default_key: str):
    t = _norm_text(user_text)
    for k, variants in mapping.items():
        if t in variants:
            return k
    return default_key

# ================== MAIN MENU ==================

if __name__ == "__main__":
    results_path = "/home/miguel-perdomo/fourier-optics/talleres/taller1/results"
    img_path = os.path.join(results_path, "Imagen_prueba_gray.png")
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(img_path)

    while True:
        print("\n=== MENÚ TALLER 1 - FOURIER ÓPTICA ===")
        print("1. Mostrar imagen original")
        print("2. Mostrar imagen muestreada (factor s)")
        print("3. Reconstrucción")
        print("0. Salir")
        opt = input("Elige una opción: ")

        if opt=="1":
            show_original(img_gray)

        elif opt=="2":
            s = int(input("Elige factor de muestreo (ej. 3,5,7): "))
            show_sampled(img_gray, s)

        elif opt=="3":
            s = int(input("Elige factor de muestreo (ej. 3,5,7): "))
            mask = np.zeros_like(img_gray, dtype=np.uint8)
            mask[::s, ::s] = 1
            sampled = (img_gray * mask)

            print("Método: 1=Fourier, 2=Convolución")
            m = input("Elige método: ").strip()

            if m=="1":
                mask_txt = input("Máscara (cuadrada/circular): ")
                recon_fourier(sampled, s, mask_txt)

            elif m=="2":
                ker_txt = input("Kernel (sinc/jinc): ")
                recon_convolution(sampled, s, ker_txt)

            else:
                print("Método no válido.")

        elif opt=="0":
            print("Saliendo...")
            break

        else:
            print("Opción no válida.")
