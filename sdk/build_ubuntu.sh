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

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# install pip
sudo apt-get install -y python3-pip

# Install python sdk
pip3 install maturin
echo 'export PATH="$PATH:$(python3 -m site --user-base)/bin"' >> ~/.bashrc
