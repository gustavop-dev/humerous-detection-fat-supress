"""
Lectura y escritura de mallas STL, sin dependencias nuevas.

Los .stl de excercies/ (húmero standard y músculos del manguito) son BINARIOS:
80 bytes de cabecera, uint32 con el número de triángulos, y 50 bytes por triángulo
(normal 3×float32, tres vértices 3×float32, uint16 de atributo).

Graphics/grafication.py ya escribe STL ASCII a mano; aquí añadimos el binario, que
es lo que necesitamos para mallas de ~40k triángulos como el subescapular.
"""

import os
import struct

import numpy as np

# Un triángulo del formato binario: normal, 3 vértices, atributo.
_STL_TRI_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("v", "<f4", (3, 3)),
    ("attr", "<u2"),
])


def _is_binary_stl(path):
    """
    Distingue binario de ASCII con la regla robusta: leer el uint32 del byte 80 y
    comprobar si el tamaño del archivo es exactamente 84 + 50·n.

    Mirar si empieza por "solid " no basta: hay exportadores que ponen esa palabra
    en la cabecera binaria.
    """
    size = os.path.getsize(path)
    if size < 84:
        return False
    with open(path, "rb") as fh:
        fh.seek(80)
        n = struct.unpack("<I", fh.read(4))[0]
    return size == 84 + 50 * n


def read_stl(path):
    """
    Lee un STL binario o ASCII.

    Devuelve (triangles, normals) con triangles (N,3,3) y normals (N,3), ambos
    float64. Las coordenadas se devuelven tal cual vienen del archivo (en estos
    datasets, milímetros).
    """
    if _is_binary_stl(path):
        with open(path, "rb") as fh:
            fh.seek(80)
            n = struct.unpack("<I", fh.read(4))[0]
            data = np.frombuffer(fh.read(50 * n), dtype=_STL_TRI_DTYPE, count=n)
        return data["v"].astype(np.float64), data["normal"].astype(np.float64)

    # ASCII: nos quedamos con las líneas 'vertex' y 'facet normal'.
    verts, norms = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "vertex" and len(parts) >= 4:
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "facet" and len(parts) >= 5:
                norms.append([float(x) for x in parts[2:5]])

    if len(verts) % 3 != 0:
        raise ValueError(f"🚨 STL ASCII malformado: {len(verts)} vértices no son múltiplo de 3")
    triangles = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    normals = (np.asarray(norms, dtype=np.float64) if len(norms) == len(triangles)
               else compute_normals(triangles))
    return triangles, normals


def compute_normals(triangles):
    """Normales por producto cruz, normalizadas. Los triángulos degenerados dan [0,0,1]."""
    tri = np.asarray(triangles, dtype=np.float64)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    bad = norm < 1e-12
    n[bad] = np.array([0.0, 0.0, 1.0])
    norm[bad] = 1.0
    return n / norm[:, None]


def write_stl_binary(path, triangles, normals=None, header=b"Geometry/stl_io"):
    """Escribe un STL binario. Si normals es None, las recalcula."""
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"🚨 se esperaba (N,3,3), llegó {tri.shape}")
    nrm = compute_normals(tri) if normals is None else np.asarray(normals, dtype=np.float64)

    rec = np.zeros(len(tri), dtype=_STL_TRI_DTYPE)
    rec["normal"] = nrm.astype("<f4")
    rec["v"] = tri.astype("<f4")

    with open(path, "wb") as fh:
        fh.write(header[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tri)))
        fh.write(rec.tobytes())
    return path


def unique_vertices(triangles, decimals=6):
    """
    Vértices únicos de la malla. Devuelve (V (M,3), faces (N,3) int).

    Deduplica redondeando: los STL repiten cada vértice en todos los triángulos que
    lo tocan, y para ajustar esfera y eje no queremos que los vértices de las zonas
    finamente malladas pesen más que los del resto.
    """
    tri = np.asarray(triangles, dtype=np.float64)
    flat = tri.reshape(-1, 3)
    _, idx, inverse = np.unique(np.round(flat, decimals), axis=0,
                                return_index=True, return_inverse=True)
    return flat[idx], inverse.reshape(-1, 3)


def triangle_areas(triangles):
    """Área de cada triángulo."""
    tri = np.asarray(triangles, dtype=np.float64)
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def sample_surface(triangles, n_points=40000, seed=0, include_vertices=True):
    """
    Muestrea puntos uniformemente sobre la SUPERFICIE de la malla (ponderando por
    área), en vez de usar solo los vértices.

    `Humero.stl` tiene 1301 vértices únicos y el 70 % están en la cabeza: la diáfisis
    se queda con ~10 vértices por banda, insuficiente para ajustar el eje y del todo
    insuficiente si además se trunca la malla. Muestrear la superficie da una nube
    densa y con densidad proporcional al área real, que es lo que hace falta para que
    los centroides por losa signifiquen algo.
    """
    tri = np.asarray(triangles, dtype=np.float64)
    areas = triangle_areas(tri)
    total = areas.sum()
    if total <= 0:
        raise ValueError("🚨 malla de área nula")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), size=int(n_points), p=areas / total)

    # Coordenadas baricéntricas uniformes sobre el triángulo.
    u = rng.random(len(idx))
    v = rng.random(len(idx))
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]

    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)

    if include_vertices:
        V, _ = unique_vertices(tri)
        pts = np.vstack([pts, V])
    return pts


def mesh_bounds(triangles):
    """(min_xyz, max_xyz) de la malla."""
    flat = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    return flat.min(axis=0), flat.max(axis=0)


def mesh_stats(path):
    """Resumen de una malla, para los chequeos de sanidad de unidades y encuadre."""
    tri, _ = read_stl(path)
    lo, hi = mesh_bounds(tri)
    V, _ = unique_vertices(tri)
    return {
        "path": path,
        "name": os.path.splitext(os.path.basename(path))[0],
        "n_triangles": int(len(tri)),
        "n_vertices": int(len(V)),
        "binary": _is_binary_stl(path),
        "bounds_min": lo,
        "bounds_max": hi,
        "extent": hi - lo,
        "centroid": V.mean(axis=0),
    }
