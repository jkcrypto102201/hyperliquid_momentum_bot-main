# Hyperliquid Trading Bot

A momentum-based algorithmic trading bot for the Hyperliquid exchange.

## AWS Server Deployment Guide

### Prerequisites
- AWS account with EC2 access
- Basic Linux command line knowledge
- Python 3.8+ experience

### Step 1: Launch EC2 Instance

1. Go to AWS EC2 Dashboard
2. Click "Launch Instance"
3. Configure:
   - Name: `hyperliquid-bot`
   - OS: Ubuntu 22.04 LTS (or latest stable)
   - Instance type: t3.small (or t3.micro for testing)
   - Storage: 20GB SSD
   - Security Group: Open ports 22 (SSH) and any monitoring ports you need
4. Launch instance and download your key pair (.pem file)

### Step 2: Connect to Your Instance

```bash
chmod 400 your-key.pem
ssh -i "your-key.pem" ubuntu@your-instance-public-ip


## Step 3: Install Dependencies

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install Excel support
sudo apt install libopenblas-dev libatlas3-base -y


## Step 4: Set Up the Trading Bot
# Clone your repository
git clone https://github.com/your-username/hyperliquid-bot.git
cd hyperliquid-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional required packages
pip install openpyxl certifi


## Step 5: Configure the Bot
cp .env.example .env
nano .env

## Add your Hyperliquid credentials:
HL_ACCOUNT_ADDRESS=your_wallet_address
HL_SECRET_KEY=your_private_key


## Create your allocations file:
cp mf_momentum_allocations.example.xlsx mf_momentum_allocations.xlsx


## Test the Bot
python main.py --test


## Step 7: Set Up as a Service (For 24/7 Operation)
sudo nano /etc/systemd/system/hyperliquid-bot.service


## Add this configuration
[Unit]
Description=Hyperliquid Trading Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/hyperliquid-bot
Environment="PATH=/home/ubuntu/hyperliquid-bot/venv/bin"
ExecStart=/home/ubuntu/hyperliquid-bot/venv/bin/python main.py

Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target


## Enable and start the service:
sudo systemctl daemon-reload
sudo systemctl enable hyperliquid-bot
sudo systemctl start hyperliquid-bot


## Check the bot's status:
sudo systemctl status hyperliquid-bot

## Verify logs
journalctl -u hyperliquid-bot -f

## Updating the bot
cd ~/hyperliquid-bot
git pull
sudo systemctl restart hyperliquid-bot

