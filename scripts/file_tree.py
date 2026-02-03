import os
import argparse
from typing import List

def generate_tree(
    root_path: str,
    prefix: str = "",
    is_last: bool = True,
    show_hidden: bool = False  # 可选：是否显示隐藏文件（以.开头）
) -> List[str]:
    """
    递归生成文件夹的树状结构字符串列表
    
    Args:
        root_path: 目标文件夹路径
        prefix: 当前层级的前缀符号（用于控制树状图格式）
        is_last: 当前条目是否是同级最后一个（控制分支符号）
        show_hidden: 是否显示隐藏文件/文件夹（Linux/macOS下以.开头，Windows下隐藏属性）
    
    Returns:
        树状结构的字符串列表，每个元素对应一行
    """
    # 初始化结果列表，首行是根目录
    tree_lines = []
    if prefix == "":
        tree_lines.append(f"📂 {os.path.basename(root_path)}/")
    
    # 获取当前目录下的所有条目（文件夹/文件），过滤隐藏文件
    try:
        entries = os.listdir(root_path)
    except PermissionError:
        tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}❌ 权限不足，无法访问")
        return tree_lines
    
    # 过滤隐藏文件（可选）
    if not show_hidden:
        entries = [e for e in entries if not e.startswith(".")]
    
    # 排序：文件夹在前，文件在后；按名称字母序排列
    entries.sort(key=lambda x: (not os.path.isdir(os.path.join(root_path, x)), x))
    
    # 遍历所有条目，生成树状结构
    for idx, entry in enumerate(entries):
        entry_path = os.path.join(root_path, entry)
        is_entry_last = idx == len(entries) - 1
        
        # 定义当前条目的前缀符号
        branch = "└── " if is_entry_last else "├── "
        # 定义下一级的前缀（控制竖线连接）
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # 判断是文件夹还是文件
        if os.path.isdir(entry_path):
            # 文件夹：加/后缀，标注📂
            tree_lines.append(f"{prefix}{branch}📂 {entry}/")
            # 递归处理子文件夹
            tree_lines.extend(generate_tree(entry_path, next_prefix, is_entry_last, show_hidden))
        else:
            # 文件：标注📄，显示文件大小（可选）
            try:
                file_size = os.path.getsize(entry_path)
                # size_str = f" ({_format_size(file_size)})"
            except:
                size_str = " (未知大小)"
            tree_lines.append(f"{prefix}{branch}📄 {entry}")
    
    return tree_lines

def _format_size(size: int) -> str:
    """辅助函数：将字节数格式化为易读的单位（B/KB/MB/GB）"""
    units = ["B", "KB", "MB", "GB"]
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units)-1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="以树状图形式输出文件夹目录结构")
    parser.add_argument(
        "folder_path",
        type=str,
        help="目标文件夹的路径（支持绝对路径/相对路径）"
    )
    parser.add_argument(
        "-s", "--show-hidden",
        action="store_true",
        help="是否显示隐藏文件/文件夹（默认不显示）"
    )
    args = parser.parse_args()
    
    # 2. 验证路径合法性
    target_path = os.path.abspath(args.folder_path)  # 转为绝对路径，避免相对路径歧义
    if not os.path.exists(target_path):
        print(f"❌ 错误：路径 '{target_path}' 不存在！")
        return
    if not os.path.isdir(target_path):
        print(f"❌ 错误：'{target_path}' 不是文件夹！")
        return
    
    # 3. 生成并打印树状图
    print(f"\n📁 文件夹目录树：{target_path}\n")
    tree_lines = generate_tree(target_path, show_hidden=args.show_hidden)
    print("\n".join(tree_lines))
    print(f"\n✅ 共 {len(tree_lines)} 个条目")

if __name__ == "__main__":
    main()
