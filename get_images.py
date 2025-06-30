from quickdraw import QuickDrawDataGroup

qdg = QuickDrawDataGroup("car", max_drawings=10)
print(qdg.drawing_count)  # how many downloaded

for idx, drawing in enumerate(qdg.drawings):
    img = drawing.image  # PIL Image
    img.convert('RGB').save(f"dataset/car/{idx}.png")
