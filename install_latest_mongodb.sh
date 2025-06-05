#!/bin/bash

# Shell script to remove existing MongoDB installation and install the latest version on Ubuntu

# Function to remove existing MongoDB installation
remove_mongodb() {
    echo "Stopping MongoDB service if running..."
    sudo systemctl stop mongod

    echo "Removing existing MongoDB packages..."
    sudo apt-get purge -y mongodb-org*
    sudo apt-get autoremove -y
    sudo apt-get autoclean

    echo "Removing any leftover MongoDB data and configuration files..."
    sudo rm -rf /var/lib/mongodb
    sudo rm -rf /etc/mongod.conf
    echo "Existing MongoDB installation removed successfully."
}

# Function to install the latest MongoDB
install_latest_mongodb() {
    echo "Updating system package database..."
    sudo apt-get update

    echo "Importing MongoDB public GPG key..."
    wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

    echo "Adding MongoDB repository to sources list..."
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

    echo "Updating package database again..."
    sudo apt-get update

    echo "Installing the latest MongoDB package..."
    sudo apt-get install -y mongodb-org

    echo "Starting MongoDB service..."
    sudo systemctl start mongod
    sudo systemctl enable mongod

    echo "MongoDB installation completed successfully."
}

# Main script execution
echo "This script will remove any existing MongoDB installation and install the latest version."
read -p "Do you wish to proceed? (yes/no): " choice

if [ "$choice" == "yes" ]; then
    remove_mongodb
    install_latest_mongodb
    echo "MongoDB has been successfully installed and started."
else
    echo "Operation canceled by the user."
    exit 1
fi