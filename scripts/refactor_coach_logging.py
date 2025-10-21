#!/usr/bin/env python3
"""
Script to refactor coach_manager_blocking.py to remove coach_log_manager
and replace with console logging + agent_feed.
"""

import re
from pathlib import Path

def refactor_file(file_path: Path):
    """Refactor coach_manager_blocking.py to remove coach_log_manager."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace coach_log_manager.append() with print()
    # Pattern: coach_log_manager.append(f"text")
    # Replace: print("text")
    
    # Simple pattern replacement
    content = re.sub(
        r'coach_log_manager\.append\(\s*f?"([^"]+)"\s*\)',
        r'print("\1")',
        content
    )
    
    # Multi-line pattern replacement
    content = re.sub(
        r'coach_log_manager\.append\(\s*\n\s*f?"([^"]+)"\s*\)',
        r'print("\1")',
        content,
        flags=re.MULTILINE
    )
    
    # Replace debug_log_manager.append() with logger.debug() or print()
    content = re.sub(
        r'debug_log_manager\.append\(\s*f?"([^"]+)"\s*\)',
        r'logger.debug("\1")',
        content
    )
    
    # Remove coach_log_manager.get_line_count() calls
    content = re.sub(
        r'coach_log_manager\.get_line_count\(\)',
        r'0  # Removed coach_log_manager',
        content
    )
    
    # Remove coach_log_manager.clear() calls
    content = re.sub(
        r'coach_log_manager\.clear\(\)',
        r'pass  # Removed coach_log_manager',
        content
    )
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Refactored: {file_path}")

if __name__ == "__main__":
    file_path = Path("/home/agile/seller_exhaustion/backtest/coach_manager_blocking.py")
    refactor_file(file_path)
