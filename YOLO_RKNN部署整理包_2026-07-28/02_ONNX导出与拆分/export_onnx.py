import shutil, os

p = r"C:\Users\baixi\Desktop"
os.makedirs(os.path.join(p, ".ultralytics"), exist_ok=True)
with open(os.path.join(p, ".ultralytics", "settings.json"), "w") as f:
    f.write("{}")

target = r"D:\python\Lib\site-packages\ultralytics\utils\__init__.py"
content = open(target, "r", encoding="utf-8").read()
content = content.replace(
    "USER_CONFIG_DIR = get_user_config_dir()",
    'USER_CONFIG_DIR = Path(r"' + os.path.join(p, '.ultralytics') + '")'
)
shutil.copyfile(target, target + ".bak")
with open(target, "w", encoding="utf-8") as f:
    f.write(content)
print("Source patched")

from ultralytics import YOLO
model_path = r"D:\py_pro\27yolo\ball2\.avls\runs\quick_train\weights\best.pt"
m = YOLO(model_path)
print(f"Model type: {m.model.yaml.get('yaml_file', 'unknown')}")
m.export(format="onnx", opset=12, simplify=True)
print("DONE - ONNX exported")
