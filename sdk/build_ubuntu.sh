# Install Ubuntu dependencies

sudo apt-get update
sudo apt-get install -y \
	build-essential \
	curl \
	clang \
	gcc \
	libssl-dev \
	llvm \
	make \
	pkg-config \
	tmux \
	xz-utils \
	ufw

sudo apt-get update
sudo apt-get install -y patchelf
sudo apt install -y pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install python sdk
python3.11 -m pip install maturin
echo 'export PATH="$PATH:$(python3.11 -m site --user-base)/bin"' >> ~/.bashrc