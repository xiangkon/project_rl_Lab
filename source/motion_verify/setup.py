from setuptools import setup, find_packages

setup(
    name="motion_verify",          # 包名（PyPI上唯一）
    version="0.1.0",            # 版本号
    author="Your Name",         # 作者
    description="A short description",  # 简介
    url="https://github.com/yourname/my_package",  # 项目主页
    packages=find_packages(),   # 自动发现所有包（替代手动列出）
    classifiers=[               # 分类信息（可选）
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",    # Python版本要求
    install_requires=[           # 依赖包（可选）
        "requests>=2.25.0",
    ],
)
