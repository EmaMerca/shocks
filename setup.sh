# references:
# pyenv https://realpython.com/intro-to-pyenv/
# poetry: https://www.adaltas.com/en/2021/06/09/pyrepo-project-initialization/


# pyenv: to switch between different python versions
apt-get install -y make build-essential libssl-dev zlib1g-dev python3-distutils \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl libarchive-dev

cd /usr/lib/x86_64-linux-gnu/
sudo ln -s -f libarchive.a liblibarchive.a && cd

curl https://pyenv.run | bash

echo '
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
' >> ~/.zshrc

exec "$SHELL"

pyenv install 3.9.11

# poetry: managing dependecies
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

source $HOME/.poetry/env

cd ~/stock-shocks
pyenv local 3.9.11
poetry new shocks
cd shocks 
poetry install 
# adding dependencies
poetry shell
poetry add --dev black flake8
poetry add numpy pandas ipykernel
# install ipykernel for notebooks

# add dependencies for libstable
sudo apt install gsl-bin libgsl0-dev


