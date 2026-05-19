#!/usr/bin/env python3
"""
Quantara: Simple Setup & Run Script

This is the ONLY file you need to:
1. Add new APIs to the system
2. Configure the project
3. Run the trading analysis

Usage:
    python run.py                    # Interactive mode
    python run.py --add-api          # Add a new API
    python run.py --query "..."      # Run with query
    python run.py --stream           # Stream results in real-time
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv, set_key

# Handle Windows UTF-8 encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantara.langgraph.main import run_langgraph_analysis, format_results


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
TOOLS_DIR = PROJECT_ROOT / "src" / "quantara" / "tools"
CORE_DIR = PROJECT_ROOT / "src" / "quantara" / "core"


# ============================================================================
# API TEMPLATES
# ============================================================================

API_TOOL_TEMPLATE = '''"""
{api_name} Tool for Quantara

This tool integrates {api_name} with the LangGraph trading system.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class {api_class_name}Tool:
    """Tool wrapper for {api_name} integration."""
    
    def __init__(self):
        """Initialize {api_name} tool."""
        self.name = "{api_name_lower}"
        self.description = "{api_description}"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute {api_name} query.
        
        Args:
            **kwargs: Query parameters
            
        Returns:
            Query results
        """
        try:
            # TODO: Implement your API logic here
            logger.info(f"Executing {self.name} with params: {{kwargs}}")
            
            result = {{
                "status": "success",
                "data": {{}},
                "message": "TODO: Implement {api_name} integration"
            }}
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {{str(e)}}")
            return {{
                "status": "error",
                "error": str(e),
                "data": {{}}
            }}


# Export for tool registry
__all__ = ["{api_class_name}Tool"]
'''

API_CORE_TEMPLATE = '''"""
{api_name} API Wrapper for Quantara

This module provides the core API integration for {api_name}.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class {api_class_name}Result:
    """Result from {api_name} API."""
    status: str
    data: Dict[str, Any]
    error: Optional[str] = None


class {api_class_name}API:
    """Core {api_name} API wrapper."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize {api_name} API.
        
        Args:
            api_key: API key for {api_name}
        """
        self.api_key = api_key
        self.base_url = "{api_base_url}"
        logger.info(f"Initialized {api_class_name}API")
    
    async def query(self, **kwargs) -> {api_class_name}Result:
        """
        Execute API query.
        
        Args:
            **kwargs: Query parameters
            
        Returns:
            {api_class_name}Result with data
        """
        try:
            # TODO: Implement your API logic here
            logger.info(f"Querying {self.base_url} with params: {{kwargs}}")
            
            return {api_class_name}Result(
                status="success",
                data={{}},
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error querying {api_class_name}: {{str(e)}}")
            return {api_class_name}Result(
                status="error",
                data={{}},
                error=str(e)
            )


# Export
__all__ = ["{api_class_name}API", "{api_class_name}Result"]
'''


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print a formatted section."""
    print(f"\n📌 {text}")
    print("-" * 40)


def check_env_file():
    """Check if .env file exists and is configured."""
    if not ENV_FILE.exists():
        print_section("⚠️  .env file not found")
        print("Creating .env from .env.example...")
        
        if ENV_EXAMPLE.exists():
            import shutil
            shutil.copy(ENV_EXAMPLE, ENV_FILE)
            print(f"✅ Created {ENV_FILE}")
        else:
            print("❌ .env.example not found!")
            return False
    
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print_section("Checking dependencies")
    
    required = [
        "langgraph",
        "langchain",
        "dotenv",
        "aiohttp",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def setup_env_key(key: str, value: str, description: str = ""):
    """Setup an environment variable."""
    print(f"\n🔑 {key}")
    if description:
        print(f"   {description}")
    
    current = os.getenv(key, "")
    if current and current != "your_key_here":
        print(f"   Current: {current[:20]}...")
        change = input("   Change? (y/n): ").strip().lower()
        if change not in ['y', 'yes']:
            return
    
    new_value = input(f"   Enter {key}: ").strip()
    if new_value:
        set_key(ENV_FILE, key, new_value)
        os.environ[key] = new_value
        print(f"   ✅ Set {key}")


def configure_apis():
    """Configure API keys interactively."""
    print_header("🔐 API CONFIGURATION")
    
    apis = {
        "OPENROUTER_API_KEY": "OpenRouter API key (required for LLM)",
        "POLYMARKET_API_KEY": "Polymarket API key (optional)",
        "KALSHI_API_KEY": "Kalshi API key (optional)",
        "APIFY_API_TOKEN": "Apify API token (optional)",
    }
    
    for key, description in apis.items():
        setup_env_key(key, "", description)
    
    print("\n✅ API configuration complete!")


def add_new_api():
    """Add a new API to the system."""
    print_header("➕ ADD NEW API")
    
    print("\nThis will create template files for a new API integration.")
    print("You'll need to implement the actual API logic.\n")
    
    api_name = input("API Name (e.g., 'CoinGecko'): ").strip()
    if not api_name:
        print("❌ API name required")
        return
    
    api_name_lower = api_name.lower()
    api_class_name = "".join(word.capitalize() for word in api_name.split())
    api_description = input("API Description: ").strip() or f"{api_name} integration"
    api_base_url = input("API Base URL (optional): ").strip() or "https://api.example.com"
    
    # Create tool file
    tool_filename = f"{api_name_lower}_tool.py"
    tool_path = TOOLS_DIR / tool_filename
    
    tool_content = API_TOOL_TEMPLATE.format(
        api_name=api_name,
        api_class_name=api_class_name,
        api_name_lower=api_name_lower,
        api_description=api_description
    )
    
    tool_path.write_text(tool_content)
    print(f"✅ Created {tool_path}")
    
    # Create core file
    core_filename = f"{api_name_lower}.py"
    core_path = CORE_DIR / core_filename
    
    core_content = API_CORE_TEMPLATE.format(
        api_name=api_name,
        api_class_name=api_class_name,
        api_base_url=api_base_url
    )
    
    core_path.write_text(core_content)
    print(f"✅ Created {core_path}")
    
    # Update tools __init__.py
    init_path = TOOLS_DIR / "__init__.py"
    init_content = init_path.read_text()
    
    import_line = f"\ntry:\n    from .{api_name_lower}_tool import *\nexcept (ImportError, ModuleNotFoundError):\n    pass\n"
    
    if f"from .{api_name_lower}_tool" not in init_content:
        init_content += import_line
        init_path.write_text(init_content)
        print(f"✅ Updated {init_path}")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Edit {tool_path}")
    print(f"   2. Edit {core_path}")
    print(f"   3. Implement your API logic")
    print(f"   4. Add API key to .env if needed")
    print(f"   5. Run: python run.py --query 'your query'")


def list_apis():
    """List all available APIs."""
    print_header("📚 AVAILABLE APIs")
    
    print("\n🔧 Core APIs:")
    core_files = sorted([f.stem for f in CORE_DIR.glob("*.py") if f.stem != "__init__"])
    for api in core_files:
        print(f"   • {api}")
    
    print("\n🛠️  Tool Wrappers:")
    tool_files = sorted([f.stem.replace("_tool", "") for f in TOOLS_DIR.glob("*_tool.py")])
    for api in tool_files:
        print(f"   • {api}")


async def run_analysis(query: str, stream: bool = False):
    """Run trading analysis."""
    print_header("🚀 QUANTARA TRADING ANALYSIS")
    
    try:
        print(f"📊 Query: {query}")
        print("⏳ Analyzing...\n")
        
        results = await run_langgraph_analysis(
            query=query,
            stream_results=stream
        )
        
        print("\n" + "=" * 60)
        print(format_results(results))
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """Run in interactive mode."""
    print_header("🤖 QUANTARA TRADING SYSTEM")
    
    while True:
        print("\n📋 MENU:")
        print("   1. Run trading analysis")
        print("   2. Configure APIs")
        print("   3. Add new API")
        print("   4. List available APIs")
        print("   5. Check dependencies")
        print("   6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            query = input("\n📝 Enter your trading query: ").strip()
            if query:
                stream = input("Stream results? (y/n): ").strip().lower() in ['y', 'yes']
                asyncio.run(run_analysis(query, stream))
        
        elif choice == "2":
            configure_apis()
        
        elif choice == "3":
            add_new_api()
        
        elif choice == "4":
            list_apis()
        
        elif choice == "5":
            check_dependencies()
        
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quantara: Trading Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Interactive mode
  python run.py --query "Find best traders"
  python run.py --query "..." --stream    # Stream results
  python run.py --add-api                 # Add new API
  python run.py --configure               # Configure APIs
  python run.py --list-apis               # List APIs
        """
    )
    
    parser.add_argument("--query", help="Run analysis with query")
    parser.add_argument("--stream", action="store_true", help="Stream results in real-time")
    parser.add_argument("--add-api", action="store_true", help="Add a new API")
    parser.add_argument("--configure", action="store_true", help="Configure API keys")
    parser.add_argument("--list-apis", action="store_true", help="List available APIs")
    parser.add_argument("--check", action="store_true", help="Check dependencies")
    
    args = parser.parse_args()
    
    # Check environment setup
    if not check_env_file():
        print("❌ Failed to setup .env file")
        return
    
    # Handle commands
    if args.add_api:
        add_new_api()
    
    elif args.configure:
        configure_apis()
    
    elif args.list_apis:
        list_apis()
    
    elif args.check:
        check_dependencies()
    
    elif args.query:
        asyncio.run(run_analysis(args.query, args.stream))
    
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
