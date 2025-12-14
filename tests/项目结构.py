import os

for root, dirs, files in os.walk(".."):
    # 跳过常见非用户代码目录
    dirs[:] = [d for d in dirs if not d.startswith(('.git', '__pycache__', 'venv', 'env', 'node_modules')) and 'site-packages' not in d]
    for file in files:
        if file.endswith('.py'):
            print(os.path.join(root, file)[2:])  # 去掉 "./"