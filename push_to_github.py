#!/usr/bin/env python3
"""
Script to push SmartNVR-GPU to GitHub as an initial commit
"""

import os
import subprocess
import sys
import getpass

def run_command(command):
    """Run a shell command and return the output"""
    try:
        print(f"Executing: {command}")
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return False

def push_to_github():
    """Push the project to GitHub"""
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Initialize git if not already done
    if not os.path.exists('.git'):
        print("\n### Initializing Git Repository ###")
        if not run_command('git init'):
            print("Error: Failed to initialize git repository.")
            return False
    
    # Check if GitHub remote is configured
    print("\n### Checking Git Remote Configuration ###")
    # Using the global subprocess import now
    result = subprocess.run(['git', 'remote', '-v'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           text=True)
    
    # Configure GitHub repository if not already done
    if "github.com/ranjanjyoti152/Smart-NVR.git" not in result.stdout:
        print("Setting up GitHub remote...")
        if not run_command('git remote remove origin'):
            print("Note: No existing origin remote to remove")
        if not run_command('git remote add origin https://github.com/ranjanjyoti152/Smart-NVR.git'):
            print("Error: Failed to add GitHub remote.")
            return False
    else:
        print("GitHub remote already configured correctly.")
    
    # Ensure the .gitignore file is respected
    print("\n### Ensuring .gitignore is respected ###")
    run_command('git rm -r --cached .')
    
    # Reset any staged changes to ensure clean state
    print("\n### Ensuring Clean State ###")
    run_command('git reset')
    
    # Add all files (respecting .gitignore)
    print("\n### Adding Files to Git ###")
    if not run_command('git add .'):
        print("Error: Failed to add files to git.")
        return False
    
    # Commit changes
    print("\n### Committing Changes ###")
    commit_message = "Initial commit: SmartNVR-GPU system with recording synchronization fixes"
    if not run_command(f'git commit -m "{commit_message}"'):
        print("Error: Failed to commit changes.")
        return False
    
    # Configure git credentials if needed
    print("\n### Configuring Git User (if needed) ###")
    # Using the global subprocess import again
    result = subprocess.run(['git', 'config', 'user.name'], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           text=True)
    
    if not result.stdout.strip():
        username = input("Enter your Git username: ")
        email = input("Enter your Git email: ")
        run_command(f'git config user.name "{username}"')
        run_command(f'git config user.email "{email}"')
    
    # Push to GitHub
    print("\n### Pushing to GitHub ###")
    if not run_command('git push -u origin master'):
        print("Error: Failed to push to GitHub. You might need to authenticate.")
        print("Try setting up a GitHub Personal Access Token:")
        print("1. Go to GitHub → Settings → Developer settings → Personal access tokens → Generate new token")
        print("2. Select 'repo' scope and create the token")
        print("3. Use the token as your password when prompted")
        
        # Try again with credential input
        print("\nAttempting push again...")
        try:
            # No need to import subprocess again
            username = input("Enter your GitHub username: ")
            token = getpass.getpass("Enter your GitHub personal access token: ")
            
            # Configure git to use credentials
            run_command(f'git remote set-url origin https://{username}:{token}@github.com/ranjanjyoti152/Smart-NVR.git')
            
            # Try pushing again
            if not run_command('git push -u origin master'):
                print("Error: Still unable to push. Check your credentials and try again manually.")
                return False
        except Exception as e:
            print(f"Error during authentication: {str(e)}")
            return False
    
    print("\nSuccess! Your code has been pushed to https://github.com/ranjanjyoti152/Smart-NVR.git")
    return True

if __name__ == "__main__":
    print("Starting GitHub push process...")
    if push_to_github():
        print("Push completed successfully!")
    else:
        print("Push failed. Please check the error messages above.")
        sys.exit(1)
