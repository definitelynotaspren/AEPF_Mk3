import os
import tarfile
import subprocess
import shutil
import urllib.request
import json

# Configuration
version = "10.2.0"
package = "Pillow"
pypi_api_url = f"https://pypi.org/pypi/{package}/{version}/json"
workdir = os.path.abspath("pillow_build_temp")
os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)

# Step 1: Query PyPI API for .tar.gz URL
print(f"Querying PyPI for {package} version {version}...")
with urllib.request.urlopen(pypi_api_url) as response:
    metadata = json.load(response)

# Step 2: Find the correct .tar.gz URL
tar_url = None
for item in metadata["urls"]:
    if item["filename"].endswith(".tar.gz"):
        tar_url = item["url"]
        break

if not tar_url:
    raise RuntimeError("Could not find source tarball URL for Pillow.")

filename = tar_url.split("/")[-1]

# Step 3: Download the tarball
print(f"Downloading: {tar_url}")
urllib.request.urlretrieve(tar_url, filename)

# Step 4: Extract it
print("Extracting archive...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()

source_dir = os.path.join(workdir, f"{package}-{version}")
setup_path = os.path.join(source_dir, "setup.py")

# Step 5: Overwrite setup.py with custom content
print("Overwriting setup.py with hardcoded version...")

custom_setup = f"""
from setuptools import setup, Extension

setup(
    name="Pillow",
    version="{version}",
    description="Python Imaging Library (Fork)",
    long_description="Hardcoded build for Python 3.13 compatibility",
    author="Alex Clark (PIL Fork Author)",
    author_email="aclark@python-pillow.org",
    url="https://python-pillow.org",
    packages=["PIL"],
    package_dir={{"": "src"}},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Historical Permission Notice and Disclaimer (HPND)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Multimedia :: Graphics :: Viewers",
    ],
    python_requires=">=3.8",
)
"""

with open(setup_path, "w", encoding="utf-8") as f:
    f.write(custom_setup.strip())
# Step 6: Build the wheel
print("Building wheel...")
subprocess.check_call([os.sys.executable, "setup.py", "bdist_wheel"], cwd=source_dir)

# Step 7: Install the wheel
dist_dir = os.path.join(source_dir, "dist")
wheel_files = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
if wheel_files:
    wheel_path = os.path.join(dist_dir, wheel_files[0])
    print(f"Installing: {wheel_path}")
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", wheel_path])
    print("✅ Pillow installed successfully!")
else:
    print("❌ No wheel file built.")
