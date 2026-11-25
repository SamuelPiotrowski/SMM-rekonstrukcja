
import open3d as o3d
from pathlib import Path

folder = Path(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY"
)

files = sorted(folder.glob("*.ply"))
print(f"Znaleziono fragmentów: {len(files)}")

geometries = []

colors = [
    [0.95, 0.35, 0.35],
    [0.35, 0.65, 0.95],
    [0.35, 0.85, 0.55],
    [0.95, 0.75, 0.30],
    [0.70, 0.45, 0.95],
    [0.95, 0.55, 0.75]
]

for i, file in enumerate(files):
    cloud = o3d.io.read_point_cloud(str(file))

    print(file.name, len(cloud.points), "punktów")

    if cloud.is_empty():
        continue

    cloud.paint_uniform_color(colors[i % len(colors)])
    geometries.append(cloud)

if geometries:
    o3d.visualization.draw_geometries(
        geometries,
        window_name="SMM 2026 - Fragmenty 3D",
        width=1200,
        height=800
    )
else:
    print("Nie udało się wczytać fragmentów.")
