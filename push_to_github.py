#!/usr/bin/env python3
"""
Push the SmartNVR-GPU project to GitHub
This script helps initialize a git repository and push to GitHub if not already done
"""
import os
import sys
import subprocess
import logging
from getpass import getpass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, capture_output=True, input_str=None):
    """Run a shell command and return the result"""
    try:
        logger.info(f"Running command: {cmd}")
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=False,
            shell=True,
            input=input_str
        )
        return result.returncode, result.stdout
    except subprocess.SubprocessError as e:
        logger.error(f"Command failed: {e}")
        return 1, str(e)

def is_git_repo():
    """Check if the current directory is already a git repository"""
    return os.path.isdir(".git")

def init_git_repo():
    """Initialize a new git repository"""
    if is_git_repo():
        logger.info("Git repository already initialized")
        return True
    
    logger.info("Initializing git repository")
    returncode, output = run_command("git init")
    return returncode == 0

def setup_git_config():
    """Setup git configuration if not already done"""
    # Check if user.name and user.email are already configured
    returncode, name = run_command("git config user.name")
    returncode, email = run_command("git config user.email")
    
    if not name.strip() or not email.strip():
        logger.info("Git user not configured. Setting up git user...")
        
        # Get user input for name and email
        git_name = input("Enter your Git username: ").strip()
        git_email = input("Enter your Git email: ").strip()
        
        if git_name and git_email:
            run_command(f'git config user.name "{git_name}"')
            run_command(f'git config user.email "{git_email}"')
            logger.info("Git user configured successfully")
            return True
        else:
            logger.error("Git user configuration failed. Name and email are required.")
            return False
    else:
        logger.info(f"Git user already configured: {name.strip()} <{email.strip()}>")
        return True

def create_gitignore():
    """Create a .gitignore file if it doesn't exist"""
    if os.path.exists(".gitignore"):
        logger.info(".gitignore file already exists")
        return
    
    logger.info("Creating .gitignore file")
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
env/
.env/

# Database files
*.db
*.sqlite*

# Logs
logs/
*.log

# Local configuration files
.env
.env.local
instance/

# Storage directories
storage/recordings/*.mp4
storage/models/
storage/thumbnails/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS generated files
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())

def add_and_commit():
    """Add all files and make initial commit"""
    logger.info("Adding all files to git")
    run_command("git add .")
    
    logger.info("Committing changes")
    returncode, output = run_command('git commit -m "Initial commit of SmartNVR-GPU project"')
    
    if "nothing to commit" in output:
        logger.info("No changes to commit")
    elif returncode == 0:
        logger.info("Changes committed successfully")
    else:
        logger.error(f"Commit failed: {output}")
        return False
    
    return True

def setup_github_remote():
    """Setup GitHub remote repository"""
    # Check if remote already exists
    returncode, output = run_command("git remote -v")
    
    if "origin" in output:
        logger.info("GitHub remote already configured")
        logger.info(f"Current remote: {output.strip()}")
        change_remote = input("Do you want to change the remote repository? (y/N): ").lower() == 'y'
        
        if not change_remote:
            return True
        
        # Remove existing remote
        run_command("git remote remove origin")
    
    # Get GitHub repository URL
    github_url = input("Enter your GitHub repository URL: ").strip()
    
    if not github_url:
        logger.error("GitHub URL is required")
        return False
    
    # Normalize GitHub URL if needed
    if not github_url.startswith("http") and not github_url.startswith("git@"):
        # Assume it's in the format username/repo
        if "/" in github_url:
            github_url = f"https://github.com/{github_url}.git"
    
    logger.info(f"Setting up GitHub remote with URL: {github_url}")
    returncode, output = run_command(f"git remote add origin {github_url}")
    
    if returncode == 0:
        logger.info("GitHub remote added successfully")
        return True
    else:
        logger.error(f"Failed to add GitHub remote: {output}")
        return False

def push_to_github():
    """Push to GitHub repository"""
    logger.info("Pushing to GitHub")
    returncode, output = run_command("git push -u origin main")
    
    # If main branch doesn't exist, try master
    if returncode != 0 and ("main" in output or "cannot find remote branch" in output):
        logger.info("Trying to push to 'master' branch instead")
        returncode, output = run_command("git push -u origin master")
    
    if returncode == 0:
        logger.info("Successfully pushed to GitHub!")
        return True
    else:
        logger.error(f"Push failed: {output}")
        
        # Check if it's an authentication issue
        if "Authentication failed" in output or "could not read Username" in output:
            logger.info("Authentication failed. Please setup your GitHub credentials.")
            logger.info("You can either:")
            logger.info("1. Configure an SSH key for GitHub")
            logger.info("2. Use the GitHub CLI to authenticate")
            logger.info("3. Use Git Credential Manager")
            logger.info("\nAfter setting up authentication, run this script again.")
        
        return False

def main():
    """Main function to push to GitHub"""
    print("\n===== SmartNVR-GPU GitHub Push Utility =====\n")
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        logger.error("This script must be run from the SmartNVR-GPU project root directory")
        return 1
    
    # Initialize git repository if needed
    if not init_git_repo():
        logger.error("Failed to initialize git repository")
        return 1
    
    # Setup git configuration
    if not setup_git_config():
        logger.error("Failed to configure git user")
        return 1
    
    # Create .gitignore file
    create_gitignore()
    
    # Add and commit files
    if not add_and_commit():
        logger.error("Failed to commit changes")
        return 1
    
    # Setup GitHub remote
    if not setup_github_remote():
        logger.error("Failed to setup GitHub remote")
        return 1
    
    # Push to GitHub
    if not push_to_github():
        logger.error("Failed to push to GitHub")
        return 1
    
    logger.info("SmartNVR-GPU project successfully pushed to GitHub!")
    logger.info("You can now access your repository on GitHub")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
