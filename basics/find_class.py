import os

site_packages = r"C:\Users\jan.petr\AppData\Local\Programs\Python\Python313\Lib\site-packages"
print(f"Searching in {site_packages}")

target_dirs = ["langchain", "langchain_community", "langchain_core", "langchain_openai"]

found = False
for target in target_dirs:
    path = os.path.join(site_packages, target)
    if not os.path.exists(path):
        continue
    
    print(f"Scanning {target}...")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                try:
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "class MultiQueryRetriever" in content:
                            print(f"FOUND IN: {full_path}")
                            found = True
                except Exception:
                    pass

if not found:
    print("MultiQueryRetriever class NOT FOUND in langchain packages.")
